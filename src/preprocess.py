def create_buildings(directory, file):
  
    #directory = '/content/drive/My Drive/Colab Notebooks/projects/UNET_100_x_100/data'
    #file = 'hkenv.txt'
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
    return buildings, polygons
