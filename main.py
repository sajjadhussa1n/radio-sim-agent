import os
import numpy as np
from src.preprocess import create_environment
from src.utils import plot_pathloss, compute_pathloss_from_fields, smooth_pathloss, compute_feature_maps
from src.nlos import compute_ci_path_loss, calc_BEL
from src.los import compute_LOS_fields
from src.reflection import compute_reflection_contributions
from src.groundref import compute_ground_reflection
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

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

tools = [simulate_radio_environment]
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


# --- 7. Example query ---
input_prompt = (
    "Generate radio simulation datasets for one random transmitter positions in each of the following "
    "locations: munich01, munich02, london, helsinki, and manhattan. "
    "For each transmitter, choose random (x, y) coordinates within the simulation area bounds, and use "
    "a UAV transmitter height between 25 m and 50 m. "
    "For every run, include all propagation components — LOS, reflections, ground reflections, NLOS, and "
    "building entry loss. Use a grid resolution of 20×20 for all simulations. "
    "Use the same random seed for reproducibility. Finally, summarize the transmitter positions, heights, "
    "and dataset file paths in a brief table at the end."
)


response = agent_executor.invoke({
    "input": input_prompt
})

print("\n=== Agent Response ===")
print(response["output"])