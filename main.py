import os
import numpy as np
from src.preprocess import create_environment
from src.utils import plot_pathloss, compute_pathloss_from_fields, smooth_pathloss, compute_feature_maps
from src.nlos import compute_ci_path_loss, calc_BEL
from src.los import compute_LOS_fields
from src.reflection import compute_reflection_contributions
from src.groundref import compute_ground_reflection
from PIL import Image
from io import BytesIO
import base64
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_core.messages import HumanMessage
import json


if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")


@tool
def simulate_radio_environment(
    tx_x: float,
    tx_y: float,
    tx_z: float,
    location: str = "helsinki",
    nx: int = 50,
    ny: int = 50,
    LOS: bool = True,
    REF: bool = True,
    GREF: bool = True,
    NLOS: bool = True,
    BEL: bool = True,
    output_dir: str = "data"
) -> str:
    """
    Runs the full radio environment simulation pipeline using deterministic 
    electromagnetic (EM) and hybrid modeling approaches. 
    
    This tool acts as the primary simulation backend for the agent, enabling 
    flexible configuration of propagation phenomena such as line-of-sight (LOS), 
    reflections, ground reflections, non-line-of-sight (NLOS) paths, and building 
    entry loss (BEL). It supports natural language-driven customization of 
    simulation parameters, including region, transmitter coordinates, and 
    spatial grid resolution.

    The simulation generates multi-layer propagation maps (LOS, REF, GREF, NLOS, BEL)
    and computes the total pathloss distribution. All intermediate results, figures, 
    and the final dataset (CSV format) are saved to the specified output directory, 
    ready for downstream operations.

    Args:
        tx_x (float): Transmitter x-coordinate in meters.
        tx_y (float): Transmitter y-coordinate in meters.
        tx_z (float): Transmitter height in meters.
        location (str, optional): Simulation region or city model to use 
            (e.g., "helsinki", "munich01", "munich02", "london", "manhattan"). Defaults to "helsinki".
        nx (int, optional): Number of horizontal grid points for the simulation 
            area. Controls spatial resolution. Defaults to 50.
        ny (int, optional): Number of vertical grid points for the simulation 
            area. Defaults to 50.
        LOS (bool, optional): Whether to include line-of-sight (LOS) computation. Defaults to True.
        REF (bool, optional): Whether to include wall or building reflection components. Defaults to True.
        GREF (bool, optional): Whether to include ground reflections. Defaults to True.
        NLOS (bool, optional): Whether to include non-line-of-sight (NLOS) contributions using 3GPP CI model. Defaults to True.
        BEL (bool, optional): Whether to include building entry loss effects for indoor receivers. Defaults to True.
        output_dir (str, optional): Directory path for saving results, plots, and dataset files. Defaults to "data".

    Returns:
        str: Summary string containing key simulation parameters, 
             and the path to the LOS, Reflection, Ground reflection and pathloss radio maps and generated dataset CSV file.

    Example:
        >>> simulate_radio_environment(
        ...     tx_x=50.0, tx_y=60.0, tx_z=20.0, 
        ...     location="helsinki", nx=128, ny=128,
        ...     LOS=True, REF=True, NLOS=False, output_dir="outputs"
        ... )
        "Simulation complete. Dataset saved at outputs/helsinki_tx50_60.csv"

    Notes:
        - Natural language input to the agent (e.g., “simulate a dense urban scene 
          with strong reflections but no NLOS”) is automatically parsed into 
          these parameters.
        - The output dataset can be directly used for UNet-based pathloss 
          prediction or hybrid training pipelines.
    """

    buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, xx, yy = create_environment(location, nx, ny)
    T = np.array([tx_x, tx_y, tx_z])  # UAV (x, y, z)
    T_horiz = T[:2]

    # Initialize Field vectors
    E_LOS = np.zeros((len(R_grid), ), dtype=np.complex128)
    E_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
    E_g_ref = np.zeros((len(R_grid), ), dtype=np.complex128)

    line_of_sight_mask = np.zeros(len(R_grid),)
    valid_reflection = np.zeros(len(R_grid),)
    E_diff = np.zeros((len(R_grid), ), dtype=np.complex128)
    E_total = np.zeros((len(R_grid), ), dtype=np.complex128)
    
    base_name = f"{location}_tx{int(tx_x)}_{int(tx_y)}_{int(tx_z)}"
    
    # 1. First, we need to compute pathloss using Ray-tracing
    if LOS:
        # Compute Direct LOS field
        E_LOS, line_of_sight_mask = compute_LOS_fields(T, R_grid, walls_array, valid_rx_mask)
        # Plot and save LOS Received Power
        P_los_map = plot_pathloss(
            E_field=E_LOS,
            xx=xx,
            yy=yy,
            walls=walls,
            T=T,
            nx=nx,
            ny=ny,
            filename=f"{base_name}_LOS.png",
            title="LOS Fields"
        )

    if REF:
        # Compute Specular Wall Reflections
        E_ref, valid_reflection = compute_reflection_contributions(R_grid, T, walls_array, walls, buildings, valid_rx_mask)
        # plot and save specular wall reflections received power 
        P_ref_map = plot_pathloss(
            E_field=E_ref,
            xx=xx,
            yy=yy,
            walls=walls,
            T=T,
            nx=nx,
            ny=ny,
            filename=f"{base_name}_reflection.png",
            title="Specular Wall Reflections"
        )

    if GREF:
        # Compute Ground Reflections
        E_g_ref = compute_ground_reflection(T, R_grid, walls_array, merged_polygons)
        # plot and save ground reflections received power
        P_r_map = plot_pathloss(
            E_field=E_g_ref,
            xx=xx,
            yy=yy,
            walls=walls,
            T=T,
            nx=nx,
            ny=ny,
            filename=f"{base_name}_ground_reflection.png",
            title="Ground Reflections"
        )

    # Compute ray-tracing pathloss
    E_total = E_LOS + E_ref + E_g_ref
    PL_total = compute_pathloss_from_fields(E_total, nx, ny)

    # 2. Now, we need to compute pathloss for NLOS points using 3GPP CI Model
    if NLOS:
        # First compute NLOS pathloss for all RX Grid
        NLOS_path_loss = compute_ci_path_loss(T, R_grid)

        # Then, points where no valid ray-tracing pathloss exist, replace it with 3gpp pathloss
        PL_total = np.where(np.isinf(PL_total), NLOS_path_loss, PL_total)

    # 3. Now, we need to introduce Building Entry Loss (BEL) in our pathloss computation
    if BEL:
        # First compute BEL for complete grid
        BEL_LOSS = calc_BEL(T, R_grid)

        # Then add BEL loss in RX locations that are inside buildings
        PL_total[~valid_rx_mask] = PL_total[~valid_rx_mask] + BEL_LOSS[~valid_rx_mask]
    

    # 4. Smooth the pathloss map
    PL = smooth_pathloss(PL_total, nx, ny)

    # plot and save Total Pathloss 
    PL_map = plot_pathloss(
        E_field=PL,
        xx=xx,
        yy=yy,
        walls=walls,
        T=T,
        nx=nx,
        ny=ny,
        cbar_label = "Pathloss (dB)",
        filename=f"{base_name}_pathloss.png",
        title="Pathloss (dB)",
        pathloss=True
    )

    # 5. Now, save Fatures Maps in CSV file

    compute_feature_maps(T, R_grid, valid_rx_mask, line_of_sight_mask, valid_reflection, buildings, PL, filename=f"{base_name}_dataset.csv")
    
    return (
        f"Radio Simulation completed successfully!\n\n"
        f"Transmitter: ({tx_x}, {tx_y}, {tx_z}) \n\n"
        f"Saved Outputs:\n"
        f" - LOS Field Map: {os.path.join(output_dir, 'LOS.png')}\n"
        f" - Reflection Map: {os.path.join(output_dir, 'reflection.png')}\n"
        f" - Ground Reflection Map: {os.path.join(output_dir, 'ground_reflection.png')}\n"
        f" - Pathloss Map: {os.path.join(output_dir, 'pathloss.png')}\n"
        f" - Dataset CSV: {os.path.join(output_dir, 'sample_pathloss_dataset_file.csv')}\n"
    )


@tool
def summarize_pathloss_image(image_path: str) -> str:
    """
    Summarizes the pathloss heatmap image (e.g., 'pathloss.png') generated by the simulation.

    The summary includes key spatial characteristics, relative power levels,
    and dominant propagation regions.
    """
    import base64
    from langchain_openai import ChatOpenAI

    if not os.path.exists(image_path):
        return f"Image not found: {image_path}"

    # Read and encode image as base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Vision-capable model
    vision_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    vision_prompt = """
You are an expert in wireless propagation and electromagnetic (EM) modeling.

Interpret the provided image as a *pathloss heatmap*, where:
- Lower dB (darker) = stronger signal (less attenuation)
- Higher dB (brighter) = weaker signal (more attenuation)

When analyzing the image, identify and describe:

1. **Approximate range** of pathloss values (in dB).  
2. **Approximate distribution** of pathloss values — e.g., uniform, skewed, clustered, or bimodal — and where each range is concentrated spatially.  
3. **Locations of strong and weak signal regions** based on the heatmap colors.  
4. **Any gradients, obstruction effects, or anomalies** such as sudden transitions near walls or reflective surfaces.

Respond concisely in 4–6 sentences, using technical language consistent with wireless propagation terminology.
"""

    # Call the model using multimodal input
    response = vision_model.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]
    )

    # Handle both string and message-object outputs safely
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    else:
        return str(response)



tools = [simulate_radio_environment, summarize_pathloss_image]
tool_names = [t.name for t in tools]


# --- 3. Model ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# --- 4. Define the prompt ---
prompt = ChatPromptTemplate.from_template("""
You are **RadioSim Agent**, an expert in deterministic EM-based
radio propagation simulations and pathloss modeling.

You can call the following tools:
{tools}

When a user provides a natural language request about radio signal behavior,
decide which tool(s) to call with appropriate parameters and interpret the results.

Follow this reasoning format:

Question: the user query
Thought: describe your reasoning
Action: select a tool to call (one of [{tool_names}])
Action Input: tool parameters (if needed)
Observation: tool output
... (you can repeat Thought/Action/Observation if needed)
Final Answer: your final summarized response to the user

Begin!

Question: {input}
{agent_scratchpad}
""")


# --- 5. Inject missing prompt variables before agent creation ---
# We provide static substitutions for {tools} and {tool_names}
prompt = prompt.partial(tools=tools, tool_names=tool_names)


# --- 6. Create agent + executor ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# input_prompt = (
#     "Run the radio simulation for location 'helsinki' at (100,100,15) with all propagation components enabled, "
#     "and then summarize the resulting pathloss heatmap png file identifying important regions."
# )





# # --- 7. Example query ---
# input_prompt = (
#     "Generate radio simulation datasets for one random transmitter positions in each of the following "
#     "locations: munich01, munich02, london, helsinki, and manhattan. "
#     "For each transmitter, choose random (x, y) coordinates within the simulation area bounds, and use "
#     "a UAV transmitter height between 25 m and 50 m. "
#     "For every run, include all propagation components — LOS, reflections, ground reflections, NLOS, and "
#     "building entry loss. Use a grid resolution of 20×20 for all simulations. "
#     "Use the same random seed for reproducibility. Finally, summarize the transmitter positions, heights, "
#     "and dataset file paths in a brief table at the end."
# )


# response = agent_executor.invoke({
#     "input": input_prompt
# })

# print("\n=== Agent Response ===")
# print(response["output"])


def evaluate_radiosim(prompts: List[str],
                      agent_executor,
                      ground_truth_params: List[Dict[str, Any]]) -> float:
    """
    Evaluate parameter extraction accuracy (PAE) of the RadioSim Agent.

    Args:
        prompts (List[str]): List of natural language simulation prompts.
        agent_executor: Initialized LangChain agent executor (RadioSim Agent).
        ground_truth_params (List[Dict[str, Any]]): List of dictionaries containing
            ground truth parameters corresponding to each prompt.

    Returns:
        float: Parameter Extraction Accuracy (PAE) — the average fraction of correctly
               extracted parameters across all prompts.
    """

    assert len(prompts) == len(ground_truth_params), \
        "Prompts and ground truth parameter lists must have the same length."

    total_correct = 0
    total_params = 0
    results = []

    print("\nEvaluating RadioSim Agent Parameter Extraction...\n")

    for i, (prompt, gt_params) in enumerate(tqdm(zip(prompts, ground_truth_params),
                                                 total=len(prompts),
                                                 desc="Processing Prompts")):
        # Build the meta prompt
        eval_prompt = (
            "Hi RadioSim Agent,\n\n"
            "You are being provided a prompt below. Read it carefully, think which tool "
            "should be called for action and extract the corresponding input parameters "
            "for that tool. **Do not call the tool** — only extract and return the parameters.\n\n"
            "Respond strictly as a Python dictionary containing parameter names and values.\n\n"
            f"Here is the prompt:\n\n{prompt}"
        )

        # Run agent to get extracted parameters
        try:
            response = agent_executor.invoke({"input": eval_prompt})
            content = response.get("output", "") if isinstance(response, dict) else str(response)

            # Attempt to parse JSON/dict-like content safely
            extracted_params = None
            try:
                extracted_params = json.loads(content)
            except json.JSONDecodeError:
                try:
                    extracted_params = eval(content)
                except Exception:
                    extracted_params = {}

            if not isinstance(extracted_params, dict):
                extracted_params = {}

        except Exception as e:
            print(f"[Warning] Error processing prompt {i}: {e}")
            extracted_params = {}

        # --- Compare extracted parameters with ground truth ---
        correct = 0
        for key, true_val in gt_params.items():
            pred_val = extracted_params.get(key, None)
            if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
                if abs(float(true_val) - float(pred_val)) < 1e-3:
                    correct += 1
            elif str(true_val).strip().lower() == str(pred_val).strip().lower():
                correct += 1

        # Record statistics
        total_correct += correct
        total_params += len(gt_params)

        results.append({
            "id": i,
            "prompt": prompt,
            "ground_truth": gt_params,
            "agent_output": extracted_params,
            "correct_count": correct,
            "total_params": len(gt_params),
            "accuracy": correct / max(1, len(gt_params))
        })

    # --- Compute overall accuracy ---
    PEA = total_correct / max(1, total_params)
    print(f"\nParameter Extraction Accuracy (PEA): {PEA:.3f}\n")

    # Save detailed log
    with open("radiosim_agent_eval_log.json", "w") as f:
        json.dump(results, f, indent=2)

    return PEA

# Example prompts and ground truth
prompts = [
    "Simulate an urban environment in Munich01 with TX at (50,60,20) and include reflections and LOS only.",
    "Run a pathloss simulation in Helsinki at (100,120,15) with all propagation components enabled."
]

ground_truth_params = [
    {"location": "munich01", "tx_x": 50.0, "tx_y": 60.0, "tx_z": 20.0, "nx":50, "ny":50, "LOS": True, "REF": True, "GREF": False, "NLOS": False, "BEL": False, "output_dir":"data"},
    {"location": "helsinki", "tx_x": 100.0, "tx_y": 120.0, "tx_z": 15.0,"nx":50, "ny":50, "LOS": True, "REF": True, "GREF": True, "NLOS": True, "BEL": True, "output_dir":"data"}
]

PAE = evaluate_radiosim(prompts, agent_executor, ground_truth_params)
