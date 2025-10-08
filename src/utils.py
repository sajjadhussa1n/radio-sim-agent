import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from scipy.ndimage import convolve
from src.constants import *

def plot_pathloss(E_field, xx, yy, walls, T, nx, ny, 
                  cbar_label = "Received Power (dBm)",
                  filename="pathloss.png", 
                  title="Pathloss Heatmap", 
                  output_dir="data", pathloss=False):
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
    if not pathloss:      
        # Power density (W/m^2)
        S = (abs(E_field) ** 2) / 377.0
    
        # Effective aperture for omni antenna (m^2)
        A_e = lambda_ ** 2 / (4 * np.pi)
    
        # Received power (W) -> dBm
        P_r = S * A_e
        P_r_dBm = 10 * np.log10(P_r) + 30
    else:
        P_r_dBm = E_field

                    
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
    PL_dBm = P_t_dBm - Pr_dBm
    
    return PL_dBm

def smooth_pathloss(pathloss, nx, ny):
    """
    Applies a 3×3 averaging filter to smooth a 2D pathloss map.

    This function performs a convolution operation where each pixel is replaced
    by the average of itself and its neighboring pixels. Edge pixels are handled
    properly by averaging only available neighbors.

    Parameters:
    -----------
    pathloss_2d : np.ndarray
        A 2D array representing pathloss values over a geographic area.

    Returns:
    --------
    np.ndarray
        A smoothed 2D pathloss map with the same shape as the input.
    """
    pathloss_2d = pathloss.reshape(ny, nx)

    # Define a 3×3 averaging kernel
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]) / 5.0

    # Apply 2D convolution
    smoothed_pathloss = convolve(pathloss_2d, kernel, mode='nearest')
    flattened_pl = smoothed_pathloss.flatten()

    return flattened_pl

def compute_feature_maps(T, R_grid, valid_rx_mask, line_of_sight_mask, valid_reflection, buildings, Path_loss, output_dir='data', filename='pathloss_dataset_file.csv'):
    
    R_horiz = R_grid[:, :2]
    T_horiz = T[:2]
    # 3D distance feature map
    Distance_3d = np.linalg.norm(R_grid - T, axis=1)
    
    dist_2d = np.linalg.norm(R_horiz - T_horiz, axis=1)
    Phi = np.arctan2(T[2], dist_2d) * (180.0 / np.pi)

    Is_building = valid_rx_mask.astype(int)

    Height_map = np.zeros_like(Is_building)
    grid_indices = [i for i in range(len(R_grid))]

    for i,rx in zip(grid_indices,R_horiz):
        for building in buildings:
            poly = building['polygon']
            height = building['height']
            if poly.covers(Point(rx)):
                Height_map[i] = height


    LOS_mask = line_of_sight_mask.astype(int)

    Reflection_mask = valid_reflection.astype(int)

    RX_X = R_grid[:,0]
    RX_Y = R_grid[:,1]
    TX_X = np.full_like(RX_X, T[0])
    TX_Y = np.full_like(RX_X, T[1])
    TX_Z = np.full_like(RX_X, T[2])

    df = pd.DataFrame({
        'RX_X': RX_X,
        'RX_Y': RX_Y,
        'TX_X': TX_X,
        'TX_Y': TX_Y,
        'TX_Z': TX_Z,
        'Distance_3d': Distance_3d,
        'Phi': Phi,
        'Is_building': Is_building,
        'height': Height_map,
        'Reflection_mask': Reflection_mask,
        'LOS_mask': LOS_mask,
        'Path_loss': Path_loss
    })

    # Root directory = parent of current file's directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(root_dir, output_dir)
    file_path = os.path.join(save_dir, file_name)
    
    # Save DataFrame to the specified directory
    df.to_csv(file_path, index=False)

    print(f"Dataset file saved successfully at: {file_path}")

    
  
