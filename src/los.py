import numpy as np
import matplotlib.pyplot as plt

def compute_facing_walls_mask(tx, walls):
    """
    Determine which walls are facing the transmitter based on their normal vectors.

    Args:
        tx (tuple): Transmitter coordinates (x, y, z).
        walls (np.ndarray): Array of walls, shape (M, 5) [x1, y1, x2, y2, height].

    Returns:
        np.ndarray: Boolean mask of shape (M,) indicating walls facing the transmitter.
    """
    tx_x, tx_y, _ = tx[0], tx[1], tx[2]

    # Extract wall endpoints
    x1 = walls[:, 0]
    y1 = walls[:, 1]
    x2 = walls[:, 2]
    y2 = walls[:, 3]
    height = walls[:, 4]

    # Create E21 as a 2D array with a third dimension of zeros
    E21 = np.stack([x2 - x1, y2 - y1, np.zeros(x1.shape)], axis=-1)

    # Compute E21_norm using broadcasting
    E21_norm = E21 / np.linalg.norm(E21, axis=-1, keepdims=True)

    E41 = np.stack([np.zeros(x1.shape), np.zeros(x1.shape), height], axis=-1)
    E41_norm = E41 / np.linalg.norm(E41, axis=-1, keepdims=True)

    wall_normal = np.cross(E21_norm, E41_norm)

    txE = np.array([])

    # Midpoint of the wall's edge
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Wall direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Outward-pointing normal vector (rotated 90 degrees counter-clockwise)
    nx = dy
    ny = -dx

    # Vector from midpoint to transmitter
    tx_mid_x = tx_x - mx
    tx_mid_y = tx_y - my

    # Dot product to check if the transmitter is in front of the wall
    dot = tx_mid_x * wall_normal[:,0] + tx_mid_y * wall_normal[:,1]
    return dot > 0

def vectorized_visibility_matrix(tx, receivers, walls, batch_size=10000):
    """
    Compute a visibility matrix of shape (N, M) where:
    - N = number of receivers
    - M = number of walls
    Each entry (i, j) is True if the line from TX to receiver i intersects wall j.
    Only walls facing the transmitter are checked for efficiency.

    Args:
        tx (tuple): Transmitter coordinates (x, y, z).
        receivers (np.ndarray): Array of receivers, shape (N, 3).
        walls (np.ndarray): Array of walls, shape (M, 5) [x1, y1, x2, y2, height].
        batch_size (int): Number of receivers processed per batch.

    Returns:
        np.ndarray: Boolean visibility matrix of shape (N, M).
    """
    tx_x, tx_y, tx_z = tx[0], tx[1], tx[2]
    N = len(receivers)
    M = len(walls)
    visibility_matrix = np.zeros((N, M), dtype=bool)

    # Identify walls facing the transmitter
    facing_walls_mask = compute_facing_walls_mask(tx, walls)
    facing_walls = walls[facing_walls_mask]
    M_facing = len(facing_walls)

    if M_facing == 0:
        return visibility_matrix  # No walls face the transmitter

    # Extract parameters for facing walls
    x1 = facing_walls[:, 0]
    y1 = facing_walls[:, 1]
    x2 = facing_walls[:, 2]
    y2 = facing_walls[:, 3]
    wall_heights = facing_walls[:, 4]
    dx_wall = x2 - x1
    dy_wall = y2 - y1

    # Process receivers in batches
    for i in range(0, N, batch_size):
        batch = receivers[i:i + batch_size]
        B = len(batch)
        rx_x = batch[:, 0][:, np.newaxis]  # Shape (B, 1)
        rx_y = batch[:, 1][:, np.newaxis]
        rx_z = batch[:, 2][:, np.newaxis]

        # Line parameters
        dx_line = rx_x - tx_x
        dy_line = rx_y - tx_y
        dz_line = rx_z - tx_z

        # Compute intersection parameters for all walls and receivers in batch
        denominator = dy_wall * dx_line - dx_wall * dy_line  # Shape (B, M_facing)
        valid = denominator != 0
        t_numerator = dx_wall * (tx_y - y1) - dy_wall * (tx_x - x1)
        t = np.divide(t_numerator, denominator, where=valid, out=np.full_like(denominator, -1))

        # Validate t and compute (x, y, z)
        t_valid = (t >= 0) & (t <= 1)
        x = tx_x + t * dx_line
        y = tx_y + t * dy_line
        z = tx_z + t * dz_line

        # Check collinearity and edge bounds
        cross = (x - x1) * dy_wall - (y - y1) * dx_wall
        collinear = np.abs(cross) < 1e-8
        dot = (x - x1) * dx_wall + (y - y1) * dy_wall
        squared_length = dx_wall**2 + dy_wall**2
        s = np.divide(dot, squared_length, where=(squared_length != 0), out=np.zeros_like(dot))
        on_edge = ((s >= 0) & (s <= 1)) | (squared_length == 0)

        # Check height
        z_valid = (z >= 0) & (z <= wall_heights)

        # Combine conditions
        intersects = valid & t_valid & collinear & on_edge & z_valid

        # Update visibility matrix for facing walls
        visibility_matrix[i:i+B, facing_walls_mask] = intersects

    return visibility_matrix

def compute_LOS_pathloss_from_Efield(distances):
    """
    Compute free-space pathloss (in dB) from the incident E-field,
    operating on a numpy array of distances.

    Parameters:
      distances   : np.array, distances from TX to RX (meters)

    Returns:
      PL          : Pathloss in dB (same shape as distances)
      E_complex   : Complex E-field (same shape as distances)

    """
    c = 3e8  # Speed of light in m/s
    f = 28e9 # Carrier frequency of 28 GHz
    lambda_ = c / f # Wave-length
    k = 2 * np.pi / lambda_  # Wave number
    omega = 2 * np.pi * f # Radian frequency
    P_t_dBm = 30  # Transmit power (dBm)
    P_t_w = 10 ** ((P_t_dBm - 30) / 10)  # Transmit power in Watts

    # Compute incident E-field magnitude (V/m);
    E = np.sqrt(30 * P_t_w) / distances

    # Compute phase component using wave number and distances
    phase = np.exp(-1j * k * distances)

    # Combine magnitude and phase to get complex E-field
    E_complex = E * phase

    # Compute power density S (W/m^2);
    S = (E ** 2) / 377.0

    # Effective aperture for an omni antenna (m^2)
    A_e = lambda_ ** 2 / (4 * np.pi)

    # Received power (W); avoid log(0) by applying a minimum floor
    P_r = S * A_e
    #P_r = np.maximum(P_r, 1e-12)
    # Convert received power to dBm\n
    P_r_dBm = 10 * np.log10(P_r) + 30


    return E_complex, P_r_dBm      

def plot_los_fields(xx, yy, los_mask, P_LOS, walls):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # LOS Visibility Plot (Binary)
    ax1 = axes[0]
    c1 = ax1.pcolormesh(xx, yy, los_mask, shading='auto', cmap='gray_r')
    ax1.set_title("LOS Visibility (1 = LOS, 0 = Blocked)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    fig.colorbar(c1, ax=ax1, label="LOS (Binary)")
    
    # FSPL Heatmap
    ax2 = axes[1]
    c2 = ax2.pcolormesh(xx, yy, P_LOS, shading='auto', cmap='viridis')
    ax2.set_title("LOS Received Power (dBm)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    fig.colorbar(c2, ax=ax2, label="Received Power (dBm)")
    
    # Superimpose the building walls on both plots
    for ax in axes:
        for wall in walls:
            x_vals = [wall['E1'][0], wall['E2'][0]]
            y_vals = [wall['E1'][1], wall['E2'][1]]
            ax.plot(x_vals, y_vals, 'k-', linewidth=2, label="Walls" if 'Walls' not in ax.get_legend_handles_labels()[1] else "")
        ax.plot(T[0], T[1], 'ro', markersize=6, label='UAV')
        ax.legend()
    
    # Display the plots
    plt.tight_layout()
    plt.show()
    
    #fig.savefig(output_directory_plots+ env_txt + 'los_plot.png', dpi=600)
