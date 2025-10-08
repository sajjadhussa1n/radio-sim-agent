import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Polygon, LineString, Point
from concurrent.futures import ProcessPoolExecutor
from shapely.ops import unary_union
import multiprocessing as mp
from src.constants import *

def check_3d_occlusion(ray_starts, ray_ends, walls, exclude_wall=None):
    """
    Vectorized 3D occlusion check between rays and vertical wall planes
    Args:
        ray_starts: (N,3) array of ray start points [x,y,z]
        ray_ends: (N,3) array of ray end points
        walls: List of wall dictionaries with keys:
            - 'E1': [x,y] start point
            - 'E2': [x,y] end point
            - 'height': float
            - 'normal': [nx, ny] (optional)
        exclude_wall: Wall to exclude from checks
    Returns:
        Boolean array indicating if ray is occluded by any wall
    """
    # Convert to numpy arrays
    ray_starts = np.asarray(ray_starts, dtype=np.float64)
    ray_ends = np.asarray(ray_ends, dtype=np.float64)

    # Filter out excluded wall
    check_walls = [w for w in walls if w is not exclude_wall]
    if not check_walls:
        return np.zeros(len(ray_starts), bool)

    # Precompute wall parameters in 3D space
    wall_params = []
    for wall in check_walls:
        E1 = np.array([*wall['E1'], 0.0], dtype=np.float64)
        E2 = np.array([*wall['E2'], 0.0], dtype=np.float64)
        height = np.float64(wall['height'])

        # Compute plane normal (perpendicular to wall in 3D)
        wall_vec = E1 - E2
        normal = np.array([-wall_vec[1], wall_vec[0], 0.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)

        wall_params.append({
            'E1': E1,
            'E2': E2,
            'height': height,
            'normal': normal,
            'plane_point': E1
        })

    # Vectorized plane intersection checks
    occluded = np.zeros(len(ray_starts), dtype=bool)

    for wall in wall_params:
        # Compute intersection parameters for all rays
        P0 = ray_starts
        D = ray_ends - ray_starts
        plane_normal = wall['normal']
        plane_point = wall['plane_point']

        # Compute denominator (N,)
        denom = D @ plane_normal

        # Avoid division by zero (parallel rays)
        valid = np.abs(denom) > 1e-12
        if not np.any(valid):
            continue

        # Compute t parameter (N,)
        t = ((plane_point - P0) @ plane_normal) / denom

        # Find valid intersections within segment
        valid_intersect = valid & (t >= 0) & (t <= 1)
        if not np.any(valid_intersect):
            continue

        # Compute intersection points (N,3)
        intersection = P0[valid_intersect] + t[valid_intersect, None] * D[valid_intersect]

        # Check vertical bounds [0, height]
        z_valid = (intersection[:, 2] >= -WALL_HEIGHT_TOLERANCE) & \
                  (intersection[:, 2] <= wall['height'] + WALL_HEIGHT_TOLERANCE)

        # Check horizontal position within wall segment
        wall_vec = wall['E2'] - wall['E1']
        wall_len = np.linalg.norm(wall_vec)
        rel_pos = intersection[:, :2] - wall['E1'][:2]
        proj = (rel_pos @ wall_vec[:2]) / wall_len

        xy_valid = (proj >= -WALL_HEIGHT_TOLERANCE) & \
                   (proj <= wall_len + WALL_HEIGHT_TOLERANCE)

        # Combine validity checks
        final_valid = z_valid & xy_valid

        # Update occlusion status
        occluded[valid_intersect] |= final_valid

    return occluded


def vectorized_occlusion_check(valid_reflection, P, R_grid, T, current_wall, all_walls, polygons):
    """3D vectorized occlusion check using vertical wall planes"""
    candidate_mask = valid_reflection.copy()
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return valid_reflection

    # Get 3D points
    P_3d = P[candidate_indices]
    R_3d = R_grid[candidate_indices]
    T_3d = np.tile(T, (len(candidate_indices), 1))

    # Check TX->P occlusions
    tx_to_p_occluded = check_3d_occlusion(T_3d, P_3d, all_walls, exclude_wall=current_wall)

    # Check P->RX occlusions
    p_to_rx_occluded = check_3d_occlusion(P_3d, R_3d, all_walls, exclude_wall=current_wall)

    # Combine results
    invalid_mask = tx_to_p_occluded | p_to_rx_occluded

    # Update valid_reflection array
    valid_reflection[candidate_indices[invalid_mask]] = False
    return valid_reflection


def process_wall(wall, the_walls, buildings, R_grid, T_horiz, T, union_excluded_dict, polygons, valid_mask):
    p_mid = wall['p_mid']
    n = wall['n']
    b_height = wall['height']
    E1 = wall['E1']
    E2 = wall['E2']
    b_id = wall['building_id']
    d = wall['length']
    this_poly = buildings[b_id]['polygon']  # building polygon of current wall
    print(f"processing {b_id}th building wall with mid-point: {p_mid}")
    R_horiz = R_grid[:, :2]

    num_rx = R_grid.shape[0]
    current_reflected_e_field = np.zeros(num_rx, dtype=np.complex128)

    # Compute the mirror image of the UAV with respect to the wall's vertical plane.
    T_mirror_horiz = T_horiz - 2 * np.dot(T_horiz - p_mid, n) * n
    T_mirror = np.array([T_mirror_horiz[0], T_mirror_horiz[1], T[2]])

    # For each receiver, parameterize the line from T_mirror to the receiver.
    R_horiz_shifted = R_horiz - T_mirror_horiz  # (num_points, 2)
    denom = R_horiz_shifted @ n  # dot product for each receiver
    eps = 1e-6
    valid_denom = np.abs(denom) > eps
    t = np.zeros(R_grid.shape[0])
    t[valid_denom] = np.dot(p_mid - T_mirror_horiz, n) / denom[valid_denom]

    valid_t = (t >= 0) & (t <= 1) & valid_denom
    t_expanded = t[:, np.newaxis]
    P = T_mirror + t_expanded * (R_grid - T_mirror)  # Intersection points (num_points x 3)
    P_horiz = P[:, :2]
    P_z = P[:, 2]

    # Check vertical constraint: P must lie between z=0 and b_height.
    valid_z = (P_z >= 0) & (P_z <= b_height)

    # Check that the horizontal intersection lies on the finite wall segment.
    d_vec = E2 - E1
    L2 = np.dot(d_vec, d_vec)
    s = np.einsum('ij,j->i', (P_horiz - E1), d_vec) / L2
    valid_seg = (s >= 0) & (s <= 1)

    valid_reflection = valid_t & valid_z & valid_seg & valid_mask

    # --- Occlusion Check ---
    union_excluded = union_excluded_dict[b_id]
    valid_reflection = vectorized_occlusion_check(valid_reflection, P, R_grid, T, wall, the_walls, polygons)
    distances_i = np.linalg.norm(P - T, axis=1)
    distances_r = np.linalg.norm(P - R_grid, axis=1)

    E_i = np.sqrt(30 * P_t_w / (distances_i ** 2))
    phase_factor_i = np.exp(-1j * k * distances_i)  # TX to reflection point
    phase_factor_r = np.exp(-1j * k * distances_r)  # Reflection point to RX

    # Complex incident field at the reflection point:
    E_i_complex = E_i * phase_factor_i

    # Unit vectors for parallel and perpendicular components of incident rays
    si = (P - T) / distances_i[:, np.newaxis]
    n_3d = np.array([n[0], n[1], 0])
    theta_i = np.arccos(np.dot(si * -1, n_3d))
    eipar = np.cross(si, np.cross(n_3d, si))
    eipar = eipar / np.linalg.norm(eipar)
    eiperp = np.cross(si, eipar)

    # Parallel and perpendicular components of reflection coefficient
    sin_theta_t = np.sqrt(epsilon0 / epsilon) * np.sin(theta_i)
    cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)

    rpar = (Zw * cos_theta_t - eta * np.cos(theta_i)) / (Zw * cos_theta_t + eta * np.cos(theta_i))
    rperp = (Zw * np.cos(theta_i) - eta * cos_theta_t) / (Zw * np.cos(theta_i) + eta * cos_theta_t)
    Gamma_squared = np.abs(rpar * rpar + rperp * rperp)

    # Parallel and perpendicular components of incident E-field
    ei_par_comp = E_i_complex[:, np.newaxis] * eipar
    ei_perp_comp = E_i_complex[:, np.newaxis] * -1.0 * eiperp

    # Parallel and perpendicular components of reflected E-field
    er_par_comp = ei_par_comp * rpar[:, np.newaxis]
    er_perp_comp = ei_perp_comp * rperp[:, np.newaxis]

    x, y, z = np.eye(3)  # Basis vectors
    result = (np.dot(eipar, x)[:, np.newaxis] * er_par_comp +
              np.dot(eipar, y)[:, np.newaxis] * er_par_comp +
              np.dot(eipar, z)[:, np.newaxis] * er_par_comp)
    result += (np.dot(eiperp, x)[:, np.newaxis] * er_perp_comp +
               np.dot(eiperp, y)[:, np.newaxis] * er_perp_comp +
               np.dot(eiperp, z)[:, np.newaxis] * er_perp_comp)

    resultant_field = result[:, 2]
    amplitude_term = distances_i / (distances_i + distances_r)
    resultant_field = resultant_field * amplitude_term * phase_factor_r

    current_reflected_e_field[valid_reflection] = resultant_field[valid_reflection]
    return current_reflected_e_field


def compute_reflection_contributions(R_grid: np.ndarray, T: np.ndarray,
                                     walls_array: np.ndarray, the_walls: list, buildings: list, valid_mask):
    """Main function with parallel processing"""
    num_rx = R_grid.shape[0]
    T_horiz = T[:2]

    # Compute visibility matrix
    visibility = vectorized_visibility_matrix(T, R_grid, walls_array, batch_size=10000)
    # First-order reflection analysis: Find walls visible to TX (at least one receiver sees it)
    reflection_walls_mask = np.any(visibility, axis=0)
    walls = [the_walls[i] for i in np.nonzero(reflection_walls_mask)[0]]

    # Precompute spatial data structures
    polygons = [b['polygon'] for b in buildings]
    union_excluded_dict = {}
    for b in buildings:
        b_id = b['building_id']
        other_polys = [p for p in polygons if p != b['polygon']]
        union_excluded_dict[b_id] = unary_union(other_polys) if other_polys else None

    # Parallel processing
    print("mp.cpu_count():", mp.cpu_count())
    num_workers = mp.cpu_count() - 1
    print("num_workers: ", num_workers)
    print(" Number of visible walls: ", len(walls))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for wall in walls:
            futures.append(executor.submit(
                process_wall, wall, the_walls, buildings, R_grid, T_horiz, T,
                union_excluded_dict, polygons, valid_mask
            ))

        total_reflected_e_field = np.zeros(num_rx, dtype=np.complex128)
        for future in futures:
            total_reflected_e_field += future.result()

    return total_reflected_e_field
