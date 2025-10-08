import os
import numpy as np
import matplotlib.pyplot as plt
from src.constants import *

def plot_pathloss(E_field, xx, yy, walls, T, nx, ny, 
                  cbar_label = "Received Power (dBm)",
                  filename="pathloss.png", 
                  title="Pathloss Heatmap", 
                  output_dir="data"):
    """
    Compute pathloss from E-field and plot the received power heatmap.

    Parameters
    ----------
    E_field : np.ndarray
        Complex or real-valued electric field array.
    xx, yy : np.ndarray
        Meshgrid arrays defining spatial coordinates.
    walls : list of dict
        Each wall is a dict with keys 'E1' and 'E2' as (x,y) tuples.
    T : tuple
        Transmitter/UAV position as (x, y).
    nx, ny : int
        Grid dimensions for reshaping.
    filename : str
        Output PNG filename (without directory path).
    title : str
        Title of the plot.
    output_dir : str
        Directory (relative to project root) where plots will be saved.
    """

    # Root directory = parent of current file's directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(root_dir, output_dir)

    # --- Pathloss Computation ---
    # Power density (W/m^2)
    S = (abs(E_field) ** 2) / 377.0

    # Effective aperture for omni antenna (m^2)
    A_e = lambda_ ** 2 / (4 * np.pi)

    # Received power (W) -> dBm
    P_r = S * A_e
    P_r_dBm = 10 * np.log10(P_r) + 30
    P_r_dBm = P_r_dBm.reshape(ny, nx)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.pcolormesh(xx, yy, P_r_dBm, shading="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(c, ax=ax, label=cbar_label)

    # Add walls
    for wall in walls:
        x_vals = [wall['E1'][0], wall['E2'][0]]
        y_vals = [wall['E1'][1], wall['E2'][1]]
        ax.plot(x_vals, y_vals, 'k-', linewidth=2,
                label="Walls" if 'Walls' not in ax.get_legend_handles_labels()[1] else "")

    # Mark UAV
    ax.plot(T[0], T[1], 'ro', markersize=6, label="UAV")
    ax.legend()

    plt.tight_layout()

    # Save figure in root/data/
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=600)
    plt.close(fig)

    print(f"Plot saved: {filepath}")

    return P_r_dBm

def compute_pathloss_from_fields(E_field, nx, ny):
    # Compute power density S (W/m^2);
    S = (abs(E_field) ** 2) / 377.0
    
    # Effective aperture for an omni antenna (m^2)
    A_e = lambda_ ** 2 / (4 * np.pi)
    
    # Received power (W); avoid log(0) by applying a minimum floor
    P = S * A_e
    # Convert received power to dBm\n
    Pr_dBm = 10 * np.log10(P) + 30
    Pr_dBm = Pr_dBm.reshape(ny,nx)
    PL_dBm = P_t_dBm - Pr_dBm
    PL_dBm = PL_total.reshape(ny,nx)
    
    return PL_dBm
