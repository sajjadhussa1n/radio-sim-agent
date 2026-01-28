import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import Transformer
import random

def extract_buildings_bbox(
    lat_min, lat_max, lon_min, lon_max,
    lat_tx, lon_tx,
    min_building_height=12, max_building_height=20
):
    """
    Extract buildings in a lat/lon bounding box, convert coordinates to meters,
    save walls in (x1 y1 x2 y2 building_id height) format,
    and plot lat/lon vs meter geometries side by side.
    """

    # ----------------------------
    # 1. Bounding polygon
    # ----------------------------
    polygon = box(lon_min, lat_min, lon_max, lat_max)

    buildings = ox.features_from_polygon(
        polygon,
        tags={"building": True}
    )

    if buildings.empty:
        print("No buildings found in the specified bounding box.")
        return

    # ----------------------------
    # 2. Coordinate transformer
    # ----------------------------
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )

    # Reference point (bottom-left of bbox)
    ref_x, ref_y = transformer.transform(lon_min, lat_min)
    ref_tx, ref_ty = transformer.traansform(lon_tx, lat_tx)
    TX_X = ref_tx - ref_x
    TX_Y = ref_ty - ref_y

    random.seed(random_seed)

    meter_polygons = []
    wall_lines = []

    # ----------------------------
    # 3. Convert buildings
    # ----------------------------
    building_id = 0
    MIN_X = 100000.0
    MAX_X = -100000.0
    MIN_Y = 100000.0
    MAX_Y = -100000.0

    for _, row in buildings.iterrows():
        geom = row.geometry

        if geom is None or geom.geom_type != "Polygon":
            continue

        # Convert polygon to meter coordinates
        coords_m = []
        for lon, lat in geom.exterior.coords:
            x, y = transformer.transform(lon, lat)
            coords_m.append((x - ref_x, y - ref_y))

        # Ensure closure
        if coords_m[0] != coords_m[-1]:
            coords_m.append(coords_m[0])

        # Random height for this building
        height = float(random.randint(min_building_height, max_building_height))

        # Convert polygon edges to wall segments
        for i in range(len(coords_m) - 1):
            x1, y1 = coords_m[i]
            x2, y2 = coords_m[i + 1]
            
            if x1 < MIN_X:
                  MIN_X = x1
            if x1 > MAX_X:
                  MAX_X = x1
            if y1 < MIN_Y:
                  MIN_Y = y1
            if y1 > MAX_Y:
                  MAX_Y = y1

            wall_lines.append(
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{building_id} {height:.2f}"
            )

        meter_polygons.append(Polygon(coords_m))
        building_id += 1

    # ----------------------------
    # 4. Save wall-based file
    # ----------------------------
    MIN_X = MIN_X - 20.0
    MIN_Y = MIN_Y - 20.0
    MAX_X = MAX_X + 20.0
    MAX_Y = MAX_Y + 20.0
    
    output_file = 'buildings.txt'
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    directory = os.path.join(root_dir, "data")
    file_path = os.path.join(directory, output_file)  
    with open(file_path, "w") as f:
        for line in wall_lines:
            f.write(line + "\n")

    print(f"Saved {building_id} buildings as wall segments.")
    print(f"Total walls written: {len(wall_lines)}")
    print(f"Height range: [{min_building_height}, {max_building_height}] meters")
    print(f"Random seed: {random_seed}")

    # ----------------------------
    # 5. Dual plot (verification)
    # ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lat/Lon plot
    buildings.plot(
        ax=axes[0],
        facecolor="lightgray",
        edgecolor="black",
        linewidth=0.5
    )
    axes[0].set_title("Buildings (Lat/Lon)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_aspect("equal")

    # (b) Meter-based plot (same style)
    for poly in meter_polygons:
        x, y = poly.exterior.xy
        axes[1].fill(
            x, y,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.5
        )

    axes[1].set_title("Buildings (Meters, EPSG:3857)")
    axes[1].set_xlabel("X (meters)")
    axes[1].set_ylabel("Y (meters)")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    # Save figure in root/data/
    filepath = os.path.join(directory, 'buildings.png')
    fig.savefig(filepath, dpi=600)
    plt.close(fig)
    #plt.show()
    return TX_X, TX_Y, MIN_X, MIN_Y, MAX_X, MAX_Y

def create_environment(MIN_X, MIN_Y, MAX_X, MAX_Y, nx=50, ny=50):

    lb = np.array([MIN_X, MIN_Y])   # left–bottom Helsinki
    rb = np.array([MAX_X, MIN_Y])   # right–bottom
    rt = np.array([MAX_X, MAX_Y])   # right–top
    lt = np.array([MIN_X, MAX_Y])   # left–top
      
    hRX = 1.50
    file = 'buildings.txt'
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    directory = os.path.join(root_dir, "data")
    file_path = os.path.join(directory, file)
    data = np.loadtxt(file_path) #  #
    
    # =============================================================================
    # Build Wall Data Structure and Group by Building
    # =============================================================================
    walls = []         # List to store each wall as a dictionary.
    buildings_dict = {}  # Group walls by building index.
    
    for row in data: 

        x1, y1, x2, y2, b_idx, h = row[0], row[1], row[2], row[3], row[4], row[5] 
      
        wall_dict = {
            'E1': np.array([x1, y1]),
            'E2': np.array([x2, y2]),
            'height': h,
            'building_id': int(b_idx)
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

    walls_array = np.zeros((len(walls), 5))
    for i, wall in enumerate(walls):
        walls_array[i, 0] = wall['E1'][0]
        walls_array[i, 1] = wall['E1'][1]
        walls_array[i, 2] = wall['E2'][0]
        walls_array[i, 3] = wall['E2'][1]
        walls_array[i, 4] = wall['height']
  
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

    xx = receiver_points[:, 0].reshape(ny, nx)
    yy = receiver_points[:, 1].reshape(ny, nx)
     
    rx_points = [Point(R_horiz[i]) for i in range(R_horiz.shape[0])]
    prepared_polys = [prep(poly) for poly in polygons]
    invalid_rx = np.zeros_like(R_horiz.shape[0], dtype=bool)
    invalid_rx = np.array([any(pp.intersects(p) for pp in prepared_polys) for p in rx_points ])
    valid_rx_mask = ~invalid_rx

    polys_list = [bld['polygon'] for bld in buildings]
    merged_polygons = unary_union(polys_list)

    return buildings, polygons, R_grid, R_horiz, valid_rx_mask, merged_polygons, walls, walls_array, xx, yy
