import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

def create_environment(file='hkenv.txt'):
  
    #directory = '/content/drive/My Drive/Colab Notebooks/projects/UNET_100_x_100/data'
    #file = 'hkenv.txt'
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    directory = os.path.join(root_dir, "data")
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path, header=None) # np.loadtxt(file_path) #  #
    
    # =============================================================================
    # Build Wall Data Structure and Group by Building
    # =============================================================================
    walls = []         # List to store each wall as a dictionary.
    buildings_dict = {}  # Group walls by building index.
    
    for i in range(len(data)): 
        row = data.iloc[i] 
        x1, y1, x2, y2, b_idx, h = row[0], row[1], row[2], row[3], row[4], row[5] 
      
        wall_dict = {
            'E1': np.array([x1, y1]),
            'E2': np.array([x2, y2]),
            'height': h,
            'building_id': b_idx
        }
        # Compute additional geometric properties.
        wall_dict['p_mid'] = (wall_dict['E1'] + wall_dict['E2']) / 2.0
        d_vec = wall_dict['E1'] - wall_dict['E2']
        L = np.linalg.norm(d_vec)
        wall_dict['length'] = L
        if L > 1e-6:
            d_unit = d_vec / L
        else:
            d_unit = np.array([0.0, 0.0])
        wall_dict['d_unit'] = d_unit
        # Candidate outward normal (rotate d_unit by +90°)
        wall_dict['n'] = np.array([-d_unit[1], d_unit[0]])
    
        walls.append(wall_dict)
    
        if b_idx not in buildings_dict:
            buildings_dict[b_idx] = []
        buildings_dict[b_idx].append(wall_dict)
    
    # =============================================================================
    # 3. Construct Building Polygons from Grouped Walls
    # =============================================================================
    # We assume that for each building the walls are given in order.
    buildings = []  # List of building dictionaries.
    polygons = []   # List of shapely Polygon objects.
    for b_idx, wall_list in buildings_dict.items():
        vertices = []
        # Use the first wall's first endpoint.
        first_wall = wall_list[0]
        vertices.append(tuple(first_wall['E1']))
        # Append the second endpoint from each wall (assumed ordered).
        for w in wall_list:
            vertices.append(tuple(w['E2']))
        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        poly = Polygon(vertices)
        buildings.append({
            'building_id': b_idx,
            'polygon': poly,
            # Use the height of the first wall as the building height.
            'height': wall_list[0]['height']
        })
        polygons.append(poly)
    
    print("Number of buildings (from file):", len(buildings))
    print("Total number of walls (from file):", len(walls))
    
  
    # Create receiver grid using bilinear interpolation.
    u_vals = np.linspace(0, 1, nx)
    v_vals = np.linspace(0, 1, ny)
    receiver_points = np.zeros((nx * ny, 2))
    index = 0
    for v in v_vals:
        for u in u_vals:
            # Bilinear interpolation:
            point = (1 - v) * ((1 - u) * lb + u * rb) + v * ((1 - u) * lt + u * rt)
            receiver_points[index, :] = point
            index += 1
    
    # Set z-coordinate for all receivers to 1.50.
    R_grid = np.column_stack((receiver_points, np.full(receiver_points.shape[0], hRX)))
    R_horiz = R_grid[:, :2]  # For horizontal computations.
     
    rx_points = [Point(R_horiz[i]) for i in range(R_horiz.shape[0])]
    prepared_polys = [prep(poly) for poly in polygons]
    invalid_rx = np.zeros_like(R_horiz.shape[0], dtype=bool)
    invalid_rx = np.array([any(pp.intersects(p) for pp in prepared_polys) for p in rx_points ])
    valid_rx_mask = ~invalid_rx

    polys_list = [bld['polygon'] for bld in buildings]
    merged_polygons = unary_union(polys_list)

    return buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons


