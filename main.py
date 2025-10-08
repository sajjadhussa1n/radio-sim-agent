import numpy as np
from src.preprocess import create_environment
from src.utils import plot_pathloss, compute_pathloss_from_fields, smooth_pathloss, compute_feature_maps
from src.nlos import compute_ci_path_loss, calc_BEL
from src.los import compute_LOS_fields
from src.reflection import compute_reflection_contributions
from src.groundref import compute_ground_reflection



buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, nx, ny, xx, yy = create_environment()
T = np.array([320, 470, 25])  # UAV (x, y, z)
T_horiz = T[:2]

# 1. First, we need to compute pathloss using Ray-tracing

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
    filename="LOS.png",
    title="LOS Fields"
)


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
    filename="reflection.png",
    title="Specular Wall Reflections"
)

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
    filename="ground_reflection.png",
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
BEL = calc_BEL(T, R_grid)

# Then add BEL loss in RX locations that are inside buildings
PL_total[~valid_rx_mask] = PL_total[~valid_rx_mask] + BEL[~valid_rx_mask]
#PL_total = PL_total.reshape(ny,nx)

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
    filename="pathloss.png",
    title="Pathloss (dB)",
    pathloss=True
)

# 5. Now, save Fatures Maps in CSV file

compute_feature_maps(T, R_grid, valid_rx_mask, line_of_sight_mask, valid_reflection, buildings, PL, filename='sample_pathloss_dataset_file.csv')











