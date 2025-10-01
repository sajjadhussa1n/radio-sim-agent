import numpy as np
from src.preprocess import create_environment
from src.los import vectorized_visibility_matrix
from src.los import compute_LOS_from_Efield
from src.los import plot_los_fields



buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, nx, ny, xx, yy = create_environment()
T = np.array([320, 470, 25])  # UAV (x, y, z)
T_horiz = T[:2]

E_LOS = np.zeros((len(R_grid), ), dtype=np.complex128)
E_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
E_g_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
E_diff = np.zeros((len(R_grid), ), dtype=np.complex128)
E_total = np.zeros((len(R_grid), ), dtype=np.complex128)

distances = np.linalg.norm(R_grid - T, axis=1)
Los_e_field, Pr_los = compute_LOS_pathloss_from_Efield(distances, valid_rx_mask)

visibility = vectorized_visibility_matrix(T, R_grid, walls_array)
line_of_sight_mask = np.all(~visibility, axis=1)
line_of_sight_mask = line_of_sight_mask & valid_rx_mask

E_LOS[line_of_sight_mask] = Los_e_field[line_of_sight_mask]

# Reshape to grid size
P_LOS = np.full_like(line_of_sight_mask, np.nan, dtype=np.float32)  # Initialize with NaN
P_LOS[line_of_sight_mask] = Pr_los[line_of_sight_mask]  # Assign valid FSPL values
P_LOS = P_LOS.reshape(ny, nx)
los_mask = line_of_sight_mask.reshape(ny, nx)

plot_los_fields(xx, yy, los_mask, P_LOS, walls)
