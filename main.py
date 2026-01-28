import os
import time
import numpy as np
from src.preprocess import extract_buildings_bbox, create_environment
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
    lat_tx: float,
    lon_tx: float,
    tx_z: float,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    min_building_height: int = 12,
    max_building_height: int = 20,
    nx: int = 50,
    ny: int = 50,
    LOS: bool = True,
    REF: bool = True,
    GREF: bool = True,
    output_dir: str = "data"
) -> str:
    """
    Executes a deterministic or hybrid electromagnetic (EM) radio environment simulation.

    This function serves as the main simulation backend for the RadioSim Agent,
    supporting both interactive simulations and evaluation-based parameter extraction.

    This function first extracts the simulation environment geometrical data within the provided latitude and longitude ranges. 

    The function then models multiple propagation components including:
    - Line-of-sight (LOS)
    - Specular reflections (REF)
    - Ground reflections (GREF)
    - Non-line-of-sight (NLOS) contributions
    - Building entry loss (BEL)

    Args:
        lat_tx (float): Transmitter Latitude.
        lon_tx (float): Transmitter Longitude.
        tx_z (float): Transmitter height (in meters).
        lat_min (float): Minimum Latitude coordinate of the simulation environment.
        lon_min (float): Minimum Longitude coordinate of the simulation environment.
        lat_max (float): Maximum Latitude coordinate of the simulation environment.
        lon_max (float): Maximum Longitude coordinate of the simulation environment.
        min_building_height (int): Minimum building height in the simulation environment. Defaults to 12.
        max_building_height (int): Minimum building height in the simulation environment. Defaults to 20.
        nx (int, optional): Grid resolution in x-direction. Defaults to 50.
        ny (int, optional): Grid resolution in y-direction. Defaults to 50.
        LOS, REF, GREF: Propagation modules to include. Defaults to True.
        output_dir (str, optional): Directory for saving simulation results. Defaults to "data".

    Returns:
        - returns a string summary with the dataset path.

    
    Example:
        >>> simulate_radio_environment(
        ...     lat_tx=51.513, lon_tx=-0.097, tx_z=20.0, 
        ...     lat_min=51.51, lon_min=-0.1, lat_max=51.516, lon_max=-0.091,
        ...     location="helsinki", nx=128, ny=128,
        ...     LOS=True, REF=True, output_dir="data"
        ... )
        "Simulation complete. Dataset saved at data/output_file.csv"
    """

    tx_x, tx_y, MIN_X, MIN_Y, MAX_X, MAX_Y = extract_buildings_bbox(lat_min, lat_max, lon_min, lon_max, lat_tx, lon_tx, min_building_height, max_building_height)
    buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, xx, yy = create_environment(MIN_X, MIN_Y, MAX_X, MAX_Y, nx=nx, ny=ny)
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
    
    base_name = f"output_tx{int(tx_x)}_{int(tx_y)}_{int(tx_z)}"
    
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
    # First compute NLOS pathloss for all RX Grid
    NLOS_path_loss = compute_ci_path_loss(T, R_grid)

    # Then, points where no valid ray-tracing pathloss exist, replace it with 3gpp pathloss
    PL_total = np.where(np.isinf(PL_total), NLOS_path_loss, PL_total)

    # 3. Now, we need to introduce Building Entry Loss (BEL) in our pathloss computation
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
        f" - LOS Field Map: {os.path.join(output_dir, f"{base_name}_LOS.png")}\n"
        f" - Reflection Map: {os.path.join(output_dir, f"{base_name}_reflection.png")}\n"
        f" - Ground Reflection Map: {os.path.join(output_dir, f"{base_name}_ground_reflection.png")}\n"
        f" - Pathloss Map: {os.path.join(output_dir, f"{base_name}_pathloss.png")}\n"
        f" - Dataset CSV: {os.path.join(output_dir, f"{base_name}_dataset.csv")}\n"
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

input_prompt = """
Task:
Simulate a radio propagation environment for a small urban area using a fixed transmitter and specified propagation mechanisms. The simulation should generate a grid-based dataset and save the results to disk.

Environment location and bounds:
Use a geographic area bounded by latitude 51.51 to 51.516 and longitude −0.10 to −0.091.

Transmitter configuration:
Place the transmitter at latitude 51.513, longitude −0.097, with a height of 20 meters above ground.

Simulation grid:
Discretize the environment into a 10 × 10 spatial grid covering the specified geographic bounds.

Propagation mechanisms:
Include line-of-sight (LOS) propagation and reflection-based (REF) propagation in the simulation. No other propagation mechanisms are required.

Output requirements:
Run the simulation using the above configuration and save the resulting dataset to the directory named "data".
When the simulation finishes successfully, confirm completion by stating that the dataset has been saved in the output directory.

Expected confirmation message:
"Simulation complete. Dataset saved at data/output_file.csv"
"""

response = agent_executor.invoke({
     "input": input_prompt
 })

print("\n=== Agent Response ===")
print(response["output"])

