import numpy as np
from src.preprocess import create_environment
from src.utils import plot_pathloss
from src.los import compute_LOS_fields
from src.reflection import compute_reflection_contributions
from src.groundref import compute_ground_reflection



buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, nx, ny, xx, yy = create_environment()
T = np.array([320, 470, 25])  # UAV (x, y, z)
T_horiz = T[:2]

E_LOS = np.zeros((len(R_grid), ), dtype=np.complex128)
E_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
E_g_ref = np.zeros((len(R_grid), ), dtype=np.complex128)
E_diff = np.zeros((len(R_grid), ), dtype=np.complex128)
E_total = np.zeros((len(R_grid), ), dtype=np.complex128)


E_LOS = compute_LOS_fields(T, R_grid, walls_array, valid_rx_mask)

LOS_map = plot_pathloss(
    E_field=E_LOS,
    xx=xx,
    yy=yy,
    walls=walls,
    T=T,
    nx=nx,
    ny=ny,
    filename="LOS_fields.png",
    title="LOS Fields"
)


# First-order reflection analysis: Find walls visible to TX (at least one receiver sees it)
#reflection_walls_mask = np.any(visibility, axis=0)
#TX_visible_walls = [walls[i] for i in np.nonzero(reflection_walls_mask)[0]]

#E_ref = compute_reflection_contributions(R_grid, T, TX_visible_walls, walls, buildings, valid_rx_mask)
#valid_reflection = E_ref != 0

#print("Reflection Fields computed successfully!")

E_g_ref = compute_ground_reflection(T, R_grid, walls_array, merged_polygons)
P_r_map = plot_pathloss(
    E_field=E_g_ref,
    xx=xx,
    yy=yy,
    walls=walls,
    T=T,
    nx=nx,
    ny=ny,
    filename="ground_reflection.png",
    title="Ground Reflection"
)




