import numpy as np
from shapely.geometry import Polygon, LineString, Point
from src.constants import *
from src.los import vectorized_visibility_matrix 

def compute_ground_reflection(T, R_grid, walls_array, merged_polygons):
    # =============================================================================
    # Ground Reflection Fields Module (Vectorized)
    # =============================================================================
    E_g_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
    num_rx = R_grid.shape[0]
    ground_reflected_e_field = np.zeros(num_rx, dtype=np.complex128)
    
    
    x_t, y_t, h_t = T[0], T[1], T[2]
    x_r, y_r, h_r = R_grid[:, 0], R_grid[:, 1], R_grid[:, 2]
    
        # Compute the reflection coefficient lambda_ref for all receivers at once
    lambda_ref = h_t / (h_t + h_r)  # Shape: (N,)
    
        # Compute ground reflection points using vectorized operations
    x_g = x_t + lambda_ref * (x_r - x_t)
    y_g = y_t + lambda_ref * (y_r - y_t)
    z_g = np.zeros_like(x_g)  # Reflection points lie on the ground (z = 0)
    
        # Stack into a (N, 3) array
    G_grid = np.column_stack((x_g, y_g, z_g))
    G_horiz = G_grid[:, :2]
    R_horiz = R_grid[:, :2]

    # Compute visibility matrix from TX to ground reflection points
    TX_to_G_visibility = vectorized_visibility_matrix(T, G_grid, walls_array, batch_size=1000)

    # Line-of-sight analysis from TX-to-ground reflection points: Find receivers with no intersections
    tx_to_g_mask = np.all(~TX_to_G_visibility, axis=1)
    
    candidate_indices = np.where(tx_to_g_mask)[0]
    for i in candidate_indices:
        seg = LineString([tuple(G_horiz[i]), tuple(R_horiz[i])])
        if seg.intersects(merged_polygons):
            tx_to_g_mask[i] = False
    
    
    distances_i = np.linalg.norm(G_grid - T, axis=1)
    distances_r = np.linalg.norm(G_grid - R_grid, axis=1)
    
    E_i = np.sqrt(30 * P_t_w / (distances_i ** 2))
    # Include propagation phase factors:
    phase_factor_i = np.exp(-1j * k * distances_i)  # TX to reflection point
    phase_factor_r = np.exp(-1j * k * distances_r)    # Reflection point to RX
    
    # Complex incident field at the reflection point:
    E_i_complex = E_i * phase_factor_i
    
    # Unit vectors for parallel and perpendicular components of incident rays
    si = (R_grid - T)/distances_i[:, np.newaxis]
    n_3d = np.array([0.0, 0.0, 1.0])
    theta_i = np.arccos(np.dot(si*-1, n_3d))
    eipar = np.cross(si, np.cross(n_3d, si))
    eipar = eipar / np.linalg.norm(eipar)
    eiperp = np.cross(si, eipar)
    
    # Parallel and perpendicular components of reflection coefficient
    sin_theta_t = np.sqrt(epsilon0 / epsilon) * np.sin(theta_i)
    cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
    
    rpar = (Zg * cos_theta_t - eta * np.cos(theta_i)) / (Zg * cos_theta_t + eta * np.cos(theta_i))
    rperp = (Zg * np.cos(theta_i) - eta * cos_theta_t) / (Zg * np.cos(theta_i) + eta * cos_theta_t)
    
            # Parallel and perpendicular components of incident E-field
    ei_par_comp = E_i_complex[:, np.newaxis] * eipar
    ei_perp_comp = E_i_complex[:, np.newaxis] * -1.0 * eiperp
    
    # Parallel and perpendicular components of reflected E-field
    er_par_comp = ei_par_comp * rpar[:, np.newaxis]
    er_perp_comp = ei_perp_comp * rperp[:, np.newaxis]
    
    x, y, z = np.eye(3)  # Basis vectors
    result = (np.dot(eipar, x)[:, np.newaxis] * er_par_comp + np.dot(eipar, y)[:, np.newaxis] * er_par_comp + np.dot(eipar, z)[:, np.newaxis] * er_par_comp)
    result += (np.dot(eiperp, x)[:, np.newaxis] * er_perp_comp + np.dot(eiperp, y)[:, np.newaxis] * er_perp_comp + np.dot(eiperp, z)[:, np.newaxis] * er_perp_comp)
    resultant_field = np.sum(result, axis=1)
    
    amplitude_term = distances_i / (distances_i + distances_r)
    resultant_field = resultant_field * amplitude_term * phase_factor_r
    
    E_g_ref[tx_to_g_mask] = resultant_field[tx_to_g_mask]

    print("Ground reflection field has been computed!!")
    return E_g_ref
