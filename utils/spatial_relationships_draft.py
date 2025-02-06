# Utils for general spatial relationships

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import ConvexHull

DEBUG=False

def find_closest_vector_key(target_vector, frames):
    # Convert the target vector to a NumPy array
    target_vector = np.array(target_vector)
    
    # Extract the vectors and keys from the frames dictionary
    keys = list(frames.keys())
    vectors = np.array(list(frames.values()))
    
    # Ensure the target vector is broadcasted properly
    target_vector = target_vector.reshape(1, -1)
    
    # Calculate the Euclidean distances using vectorized operations
    distances = np.linalg.norm(vectors - target_vector, axis=1)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # Return the key of the closest vector
    return keys[closest_index]

def get_boxes_relationship(obj1, obj2, extrinsic, intrinsic):
    """Get the spatial relationship of two 3D boxes on the image and in world coordinates.

    Args:
        obj1 (dict): First object with 'name' and 'box' (OrientedBoundingBox).
        obj2 (dict): Second object with 'name' and 'box' (OrientedBoundingBox).
        extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world
            transformation.
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.

    Returns:
        tuple: Spatial relationship (horizontal, depth, vertical_world, object_centric_horizontal, object_centric_depth) of the two boxes.
    """
    def get_box2d_coordinates_and_depth(box, extrinsic, intrinsic):
        extrinsic_w2c = np.linalg.inv(extrinsic)
        corners = np.asarray(box.get_box_points())
        corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
        corners = np.concatenate(
            [corners, np.ones((corners.shape[0], 1))], axis=1)
        corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
        corners_img = corners_img.transpose()
        corners_pixel = np.zeros((corners_img.shape[0], 2))
        corners_depth = np.zeros(corners_img.shape[0])

        for i in range(corners_img.shape[0]):
            corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])
            corners_depth[i] = corners_img[i][2]

        return corners_pixel.astype(int).tolist(), corners_depth.tolist()

    def determine_camera_and_world_relationship(box1_coords, box2_coords, box1_depth, box2_depth, box1_world_coords, box2_world_coords):
        box1_x_coords = [coord[0] for coord in box1_coords]
        box1_y_coords = [coord[1] for coord in box1_coords]
        box2_x_coords = [coord[0] for coord in box2_coords]
        box2_y_coords = [coord[1] for coord in box2_coords]

        if all(x1 < x2 for x1 in box1_x_coords for x2 in box2_x_coords):
            horizontal_relation = "left"
        elif all(x1 > x2 for x1 in box1_x_coords for x2 in box2_x_coords):
            horizontal_relation = "right"
        else:
            horizontal_relation = "overlapping"

        if all(d1 < d2 for d1 in box1_depth for d2 in box2_depth):
            depth_relation = "in front of"
        elif all(d1 > d2 for d1 in box1_depth for d2 in box2_depth):
            depth_relation = "behind"
        else:
            depth_relation = "overlapping"

        box1_world_z_coords = [coord[2] for coord in box1_world_coords]
        box2_world_z_coords = [coord[2] for coord in box2_world_coords]

        if all(z1 < z2 for z1 in box1_world_z_coords for z2 in box2_world_z_coords):
            vertical_world_relation = "below"
        elif all(z1 > z2 for z1 in box1_world_z_coords for z2 in box2_world_z_coords):
            vertical_world_relation = "above"
        else:
            vertical_world_relation = "overlapping"


        return {
            "left": True if horizontal_relation == "left" else False,
            "right": True if horizontal_relation == "right" else False,
            "infront": True if depth_relation == "in front of" else False,
            "behind": True if depth_relation == "behind" else False,
            "cam_overlapping": True if horizontal_relation == "overlapping" else False,
            "above": True if vertical_world_relation == "above" else False,
            "below": True if vertical_world_relation == "below" else False,
            "world_overlapping": True if vertical_world_relation == "overlapping" else False
        }

        return horizontal_relation, depth_relation, vertical_world_relation

    def get_object_centric_relationship(obj1, obj2):
        def get_facing_direction(box):
            rotation_matrix = np.asarray(box.R)
            forward_direction = rotation_matrix[:, 0]
            return forward_direction

        def check_overlap(box1, box2):
            box1_points = np.asarray(box1.get_box_points())
            box2_points = np.asarray(box2.get_box_points())

            def project_points(points, axis):
                return np.dot(points, axis)

            def overlap_on_axis(box1_proj, box2_proj):
                box1_min, box1_max = np.min(box1_proj), np.max(box1_proj)
                box2_min, box2_max = np.min(box2_proj), np.max(box2_proj)
                return not (box1_max < box2_min or box2_max < box1_min)

            axes = np.vstack((np.diff(box1_points, axis=0), 
                              np.diff(box2_points, axis=0))).reshape(-1, 3)
            for axis in axes:
                if not overlap_on_axis(project_points(box1_points, axis),
                                       project_points(box2_points, axis)):
                    return False, False, False

            return True, True, True

        overlap_x, overlap_y, overlap_z = check_overlap(obj1["box"], obj2["box"])

        obj2_forward = get_facing_direction(obj2["box"])
        obj1_center = np.mean(np.asarray(obj1["box"].get_box_points()), axis=0)

        relative_position = obj1_center - np.asarray(obj2["box"].get_center())
        dot_product = np.dot(relative_position, obj2_forward)
        cross_product = np.cross(obj2_forward, relative_position)

        if overlap_x and overlap_y and not overlap_z:
            depth_relation = "overlapping"
        elif dot_product > 0:
            depth_relation = "in front of"
        else:
            depth_relation = "behind"

        if overlap_x or overlap_y:
            horizontal_relation = "overlapping"
        elif cross_product[2] > 0:
            horizontal_relation = "left"
        else:
            horizontal_relation = "right"

        return horizontal_relation, depth_relation #NOTE Horizontal relationship is subjective so don't use it

    box1_coords, box1_depth = get_box2d_coordinates_and_depth(obj1["box"], extrinsic, intrinsic)
    box2_coords, box2_depth = get_box2d_coordinates_and_depth(obj2["box"], extrinsic, intrinsic)

    box1_world_coords = np.asarray(obj1["box"].get_box_points()).tolist()
    box2_world_coords = np.asarray(obj2["box"].get_box_points()).tolist()

    cam_world_relations = determine_camera_and_world_relationship(
        box1_coords, box2_coords, box1_depth, box2_depth, box1_world_coords, box2_world_coords)

    obj_centric_horizontal, obj_centric_depth = get_object_centric_relationship(obj1, obj2)

    relationships = {
        "camera_centric": {
            "left": cam_world_relations["left"],
            "right": cam_world_relations["right"],
            "infront": cam_world_relations["infront"],
            "behind": cam_world_relations["behind"],
            "above": cam_world_relations["above"],
            "below": cam_world_relations["below"],
            "overlapping": cam_world_relations["cam_overlapping"]
        },
        "world_centric": {
            "left": cam_world_relations["left"],
            "right": cam_world_relations["right"],
            "infront": cam_world_relations["infront"],
            "behind": cam_world_relations["behind"],
            "above": cam_world_relations["above"],
            "below": cam_world_relations["below"],
            "overlapping": cam_world_relations["world_overlapping"]
        },
        "object_centric": {
            "left": obj_centric_horizontal == "left",
            "right": obj_centric_horizontal == "right",
            "infront": obj_centric_depth == "in front of",
            "behind": obj_centric_depth == "behind",
            "above": cam_world_relations["above"],
            "below": cam_world_relations["below"],
            "overlapping": obj_centric_horizontal == "overlapping" or obj_centric_depth == "overlapping"
        }
    }

    return relationships



# Function to project the bounding box to the floor (2D)
def project_to_floor(box):
    corners = np.asarray(box.get_box_points())
    # corners[:, 2] = 0  # Set the z-coordinate to 0 to project onto the floor
    return corners[:, :2]  # Return only the x, y coordinates

# Function to create the grid representing the floor
def create_floor_grid(floor_box, grid_resolution=0.1):

    min_bound = floor_box.get_min_bound()[:2]
    max_bound = floor_box.get_max_bound()[:2]
    
    x_range = np.arange(min_bound[0], max_bound[0], grid_resolution)
    y_range = np.arange(min_bound[1], max_bound[1], grid_resolution)
    
    return np.meshgrid(x_range, y_range)

# # Function to mark occupied areas on the grid
# def mark_occupied_areas(grid, boxes):
#     occupied = np.zeros(grid[0].shape, dtype=bool)
#     for box in boxes:
#         projected_points = project_to_floor(box)
#         hull = o3d.geometry.ConvexHull.create_from_points(o3d.utility.Vector3dVector(projected_points))

#         # Find the index of the minimum and maximum points
#         min_index = np.argmin(corners, axis=0)
#         max_index = np.argmax(corners, axis=0)

#         # Retrieve the actual minimum and maximum points
#         x_min, y_min = corners[min_index]
#         x_max, y_max = corners[max_index]

#         print((x_min, y_min), (x_max, y_max))
        
#         x_indices = (grid[0] >= x_min) & (grid[0] <= x_max)
#         y_indices = (grid[1] >= y_min) & (grid[1] <= y_max)
        
#         occupied |= x_indices & y_indices
    
#     return occupied

# Function to mark occupied areas on the grid
def mark_occupied_areas(grid, boxes, occupied, floor=False):
    for box in boxes:
        projected_points = project_to_floor(box)
        hull = ConvexHull(projected_points)

        hull_vertices = projected_points[hull.vertices]
        path = mpath.Path(hull_vertices)

        ## Check each grid point if it is inside the hull
        for idx in np.ndindex(grid[0].shape):
            i, j = idx
            point = (grid[0][i, j], grid[1][i, j])
            if floor:
                if not path.contains_point(point):
                    occupied[i, j] = True
            elif path.contains_point(point):
                occupied[i, j] = True
        
    return occupied

# Function to find empty areas on the grid
def find_empty_areas(grid, occupied):
    empty_areas = np.logical_not(occupied)
    return empty_areas

def get_empty_space(floor_box, boxes, grid_resolution=0.1):
    grid = create_floor_grid(floor_box, grid_resolution)
    empty_occupied = np.zeros(grid[0].shape, dtype=bool)
    floor_occupied = mark_occupied_areas(grid, [floor_box], empty_occupied, floor=True)
    occupied = mark_occupied_areas(grid, boxes, floor_occupied)
    empty_areas = find_empty_areas(grid, occupied)

    # if DEBUG:
    #     # Plot the grid to visualize
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(empty_areas, extent=[floor_box.get_min_bound()[0], floor_box.get_max_bound()[0], floor_box.get_min_bound()[1], floor_box.get_max_bound()[1]], origin='lower', cmap='gray')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Empty Areas on the Floor')
    #     plt.show()

    return empty_areas, grid, occupied

def box_intersection(box1, box2):
    box1_corners = np.asarray(box1.get_box_points())
    box2_corners = np.asarray(box2.get_box_points())
    
    box1_min = box1_corners.min(axis=0)
    box1_max = box1_corners.max(axis=0)
    box2_min = box2_corners.min(axis=0)
    box2_max = box2_corners.max(axis=0)
    
    return np.all(box1_max >= box2_min) and np.all(box2_max >= box1_min)

def can_fit_on_top(top_box, base_box):
    """Determines if the top OrientedBoundingBox can fit on top of the base OrientedBoundingBox.
        
    Args:
        top_box (o3d.geometry.OrientedBoundingBox): The top bounding box that needs to fit on the base box.
        base_box (o3d.geometry.OrientedBoundingBox): The base bounding box.
        
    Returns:
        bool: True if the top box can fit on top of the base box, False otherwise.
    """
    # Get the dimensions of the bounding boxes
    base_extent = base_box.extent
    top_extent = top_box.extent
    
    # Check if the top box can fit within the base box's x and y dimensions
    if (top_extent[0] <= base_extent[0] and
        top_extent[1] <= base_extent[1]):
        return True
    
    return False


# def can_fit(occupied, box_corners, grid, grid_resolution=0.1):
#     x_min, y_min = box_corners.min(axis=0)
#     x_max, y_max = box_corners.max(axis=0)
    
#     x_indices = (grid[0] >= x_min) & (grid[0] <= x_max)
#     y_indices = (grid[1] >= y_min) & (grid[1] <= y_max)
    
#     region = x_indices & y_indices
#     return not occupied[region].any()




def can_relocate_object(env_grid, resolution, boxes, floor_box, moving_box, anchor_box):
    # Function to get the 2D projection of the bounding box on the floor
    def get_2d_projection(bbox):
        points = np.asarray(bbox.get_box_points())
        floor_points = points[:, [0, 2]]  # We consider x and z coordinates for 2D floor projection
        return floor_points

    # Convert world coordinates to grid indices
    def world_to_grid(point, resolution):
        return np.round(point / resolution).astype(int)

    # Check if all points in the moving box are within empty space in the grid
    def is_space_empty(moving_box_2d, direction_vector, anchor_center, threshold_distance):
        for point in moving_box_2d:
            translated_point = point + direction_vector * threshold_distance
            grid_point = world_to_grid(translated_point, resolution)
            if not env_grid[grid_point[0], grid_point[1]]:
                return False
        return True
    
    # Get the 2D projections of the moving box and anchor box
    moving_box_2d = get_2d_projection(moving_box)
    anchor_box_2d = get_2d_projection(anchor_box)

    # Calculate the area in front of the anchor box where the moving box should be relocated
    # Here we assume "in front" means in the positive x-direction relative to the anchor box's orientation
    anchor_center = np.mean(anchor_box_2d, axis=0)
    moving_center = np.mean(moving_box_2d, axis=0)
    direction_vector = moving_center - anchor_center
    direction_vector /= np.linalg.norm(direction_vector)

    # Define a threshold distance in front of the anchor box to check
    threshold_distance = np.linalg.norm(anchor_box.extent)
    
    # Check if the potential position is within the empty space and large enough
    if is_space_empty(moving_box_2d, direction_vector, anchor_center, threshold_distance):
        return True
    return False

def can_fit_object_a_in_front_of_b(floor_box, environment_boxes, object_a, object_b, grid_resolution=0.1):
    # Get the empty space on the floor
    empty_areas, grid, occupied = get_empty_space(floor_box, environment_boxes, 0.1)
    
    # Calculate the front direction of object_b
    front_direction = object_b.R[:, 0]  # Assuming the front direction is along the x-axis of object_b

    # Define the bounding box dimensions of object_a
    a_extent = object_a.extent[:2]  # Only consider x and y dimensions
    
    # Iterate over the grid to check all areas in front of object_b
    for i in range(empty_areas.shape[0]):
        for j in range(empty_areas.shape[1]):
            if empty_areas[i, j]:
                # Calculate the potential center position for object_a
                x, y = grid[0][i, j], grid[1][i, j]
                
                # Check if object_a fits within the grid cell
                if not empty_areas[max(0, i-int(a_extent[0]/grid_resolution)):min(empty_areas.shape[0], i+int(a_extent[0]/grid_resolution)),
                                   max(0, j-int(a_extent[1]/grid_resolution)):min(empty_areas.shape[1], j+int(a_extent[1]/grid_resolution))].all():
                    continue
                
                placement_center = np.array([x, y, object_b.center[2]])
                placed_object_a = o3d.geometry.OrientedBoundingBox(placement_center, object_a.R, object_a.extent)
                
                # Check intersection with environment
                collision_detected = False
                for box in environment_boxes:
                    if box_intersection(placed_object_a, box):
                        collision_detected = True
                        break
                
                if not collision_detected:
                    return True

    return False



def can_fit_object_a_in_relation_to_b(floor_box, environment_boxes, box_a, box_b, extrinsic, intrinsic, threshold=0.1, grid_resolution=0.1):
    """
    Check if object A can fit in the specified positions (left, right, in front, behind) relative to object B
    in both the camera frame and the object frame.

    Args:
        box_a (o3d.geometry.OrientedBoundingBox): Object A's bounding box.
        box_b (o3d.geometry.OrientedBoundingBox): Object B's bounding box.
        extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world transformation.
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.

    Returns:
        dict: Dictionary indicating if object A can fit in all possible positions relative to object B
              in both the camera frame and the object frame.
    """
    def project_to_floor(box):
        corners = np.asarray(box.get_box_points())
        floor_corners = corners[:, :2]  # Projecting to floor (x, y plane)
        return floor_corners

    def can_fit_old(occupied, box_corners, grid_x, grid_y, grid_resolution):
        for corner in box_corners:
            grid_x_idx = int((corner[0] - grid_x.min()) / grid_resolution)
            grid_y_idx = int((corner[1] - grid_y.min()) / grid_resolution)
            if occupied[grid_x_idx, grid_y_idx]:
                return False
        return True
    
    def can_fit(grid, box, occupied):
        bbox = compute_oriented_bounding_box(box)
        projected_points = project_to_floor(bbox)
        hull = ConvexHull(projected_points)

        hull_vertices = projected_points[hull.vertices]
        path = mpath.Path(hull_vertices)

        ## Check each grid point if it is inside the hull
        for idx in np.ndindex(grid[0].shape):
            i, j = idx
            point = (grid[0][i, j], grid[1][i, j])
            if path.contains_point(point):
                if occupied[i, j]:
                    return False
        return True

    # Get the empty space on the floor
    empty_areas, grid, occupied = get_empty_space(floor_box, environment_boxes, 0.1)

    # Move the Box A to Box B position
    # Calculate the translation needed to move obb1's center to obb2's center
    translation = box_b.center - box_a.center

    # Translate obb1
    box_a.translate(translation)

    # Project Box A and Box B to floor
    box_a_corners = project_to_floor(box_a)
    box_b_corners = project_to_floor(box_b)

    results = {}

    # Check positions in both frames
    frames = {
        'objectcentric': {
            'infront_direction': np.asarray(box_b.R)[:, 0],  # Box B's facing direction
            'left_direction': np.asarray(box_b.R)[:, 1],  # Perpendicular to the facing direction (left direction)
            'behind_direction': -1 * np.asarray(box_b.R)[:, 0],  # Box B's facing direction
            'right_direction': -1 * np.asarray(box_b.R)[:, 1]  # Perpendicular to the facing direction (left direction)
        },
        'cameracentric' : {}
        # 'cameracentric': {
        #     'forward_direction': -1 * np.asarray(extrinsic[:, :3][:, 2]),  # Camera's forward direction (z-axis)
        #     'left_direction': np.asarray(-1 * extrinsic[:3, 0])  # Camera's left direction (x-axis)
        # }
    }


    camera_infront = find_closest_vector_key(np.asarray(-1 * extrinsic[:, :3][:, 2][:3]), frames["objectcentric"])
    camera_behind = find_closest_vector_key(np.asarray(extrinsic[:, :3][:, 2][:3]),  frames["objectcentric"])
    camera_left = find_closest_vector_key(np.asarray(-1 * extrinsic[:3, 0]),  frames["objectcentric"])
    camera_right = find_closest_vector_key(np.asarray(extrinsic[:3, 0]),  frames["objectcentric"])


    frames['cameracentric']["infront_direction"] = {camera_infront: frames['objectcentric'][camera_infront]}
    frames['cameracentric']["behind_direction"] = {camera_behind: frames['objectcentric'][camera_behind]}
    frames['cameracentric']["left_direction"] = {camera_left: frames['objectcentric'][camera_left]}
    frames['cameracentric']["right_direction"] = {camera_right: frames['objectcentric'][camera_right]}

    for frame, directions in frames.items():
        # if frame == "cameracentric":
        #     continue


        # print(extrinsic)
        # print(np.asarray(extrinsic[:, :3][:, 2]))

        # print(box_b.extent[0], box_b.extent[1])
        # print(box_a.extent[0], box_a.extent[1])


        translation_positions = {
            "left_direction": (box_b.extent[1] / 2 + box_a.extent[1] / 2 + threshold),
            "right_direction": (box_b.extent[1] / 2 + box_a.extent[1] / 2 + threshold),
            "infront_direction": (box_b.extent[0] / 2 + box_a.extent[0] / 2 + threshold),
            "behind_direction": (box_b.extent[0] / 2 + box_a.extent[0] / 2 + threshold)
        }

        results[frame] = {}

        for direction, value in directions.items():

            # if direction not in ["left_direction", "right_direction", "behind_direction"]:
            #     continue
            
            if frame == "cameracentric":
                actual_direction = list(value.keys())[0]
                translation = translation_positions[actual_direction]
                translated_box_a_corners = box_a_corners + translation * value[actual_direction][:2]

            else:
                translation = translation_positions[direction]
                translated_box_a_corners = box_a_corners + translation * value[:2]



            # print(translated_box_a_corners)
            # print(direction, frame)
            # bbox = compute_oriented_bounding_box(translated_box_a_corners)

            # occupied = mark_occupied_areas(grid, [bbox], occupied)
            # empty_areas = find_empty_areas(grid, occupied)
            results[frame][direction] = can_fit(grid, translated_box_a_corners, occupied)

            if DEBUG:
                # Place the box
                debug_occupied = occupied.copy()
                debug_grid = grid.copy()
                bbox = compute_oriented_bounding_box(translated_box_a_corners)
                debug_occupied = mark_occupied_areas(debug_grid, [bbox], debug_occupied)
                debug_empty_areas = find_empty_areas(debug_grid, debug_occupied)

                # Plot the grid to visualize
                plt.figure(figsize=(10, 10))
                plt.imshow(debug_empty_areas, extent=[floor_box.get_min_bound()[0], floor_box.get_max_bound()[0], floor_box.get_min_bound()[1], floor_box.get_max_bound()[1]], origin='lower', cmap='gray')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'{frame} {direction} {results[frame][direction]}')
                plt.show()

    return results

def another_find_positions_for_box_a(floor_box, boxes, box_a, box_b, extrinsic, intrinsic, grid_resolution=0.1):
    grid = create_floor_grid(floor_box)
    occupied = mark_occupied_areas(grid, boxes)
    box_b_corners = project_to_floor(box_b)
        
    possible_positions = {
        "world_centric": {},
        "object_centric": {}
    }
    
    for position in ["left", "right", "infront", "behind"]:
        dx, dy = 0, 0
        if position == "left":
            dx = -box_b.extent[0]
        elif position == "right":
            dx = box_b.extent[0]
        elif position == "infront":
            dy = box_b.extent[1]
        elif position == "behind":
            dy = -box_b.extent[1]
        
        translation = np.array([dx, dy])
        box_a_corners = project_to_floor(box_a) + translation
        
        if can_fit(occupied, box_a_corners, grid, grid_resolution):
            possible_positions["world_centric"][position] = True
        else:
            possible_positions["world_centric"][position] = False

    forward_direction = np.asarray(box_b.R)[:, 0]  # Box B's facing direction
    left_direction = np.asarray(box_b.R)[:, 1]  # Perpendicular to the facing direction (left direction)
    
    for position in ["left", "right", "infront", "behind"]:
        translation = np.array([0, 0])
        if position == "left":
            translation = -box_b.extent[0] * left_direction[:2]
        elif position == "right":
            translation = box_b.extent[0] * left_direction[:2]
        elif position == "infront":
            translation = box_b.extent[1] * forward_direction[:2]
        elif position == "behind":
            translation = -box_b.extent[1] * forward_direction[:2]
        
        box_a_corners = project_to_floor(box_a) + translation
        
        if can_fit(occupied, box_a_corners, grid, grid_resolution):
            possible_positions["object_centric"][position] = True
        else:
            possible_positions["object_centric"][position] = False

    return possible_positions

# Function to determine if Box A can fit relative to Box B
def test_find_positions_for_box_a(floor_box, boxes, box_a, box_b, grid_resolution=0.1):
    grid = create_floor_grid(floor_box)
    occupied = mark_occupied_areas(grid, boxes)
    box_b_corners = project_to_floor(box_b)

    # Relative positions
    positions = {
        'left': (-box_b.extent[0], 0),
        'right': (box_b.extent[0], 0),
        'infront': (0, box_b.extent[1]),
        'behind': (0, -box_b.extent[1]),
    }
    
    for position, (dx, dy) in positions.items():
        translation = np.array([dx, dy])
        box_a_corners = project_to_floor(box_a) + translation
        
        if can_fit(occupied, box_a_corners, grid, grid_resolution):
            return position
    
    return None



def compute_oriented_bounding_box(points_2d):
    """
    Computes an oriented bounding box from a 2D array of x, y points.

    Parameters:
    points_2d (numpy.ndarray): A 2D array of shape (N, 2) representing x, y coordinates of N points.

    Returns:
    open3d.geometry.OrientedBoundingBox: The oriented bounding box of the input points.
    """
    # Convert 2D points to 3D by adding a z-coordinate of 0
    points_3d = np.c_[points_2d, np.zeros(points_2d.shape[0])]

    # Slightly perturb the z-coordinates to avoid coplanarity issues
    points_3d[:, 2] += np.random.uniform(low=-1e-5, high=1e-5, size=points_3d.shape[0])

    # Create an open3d point cloud from the 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # Compute the oriented bounding box
    oriented_bbox = point_cloud.get_oriented_bounding_box()

    return oriented_bbox