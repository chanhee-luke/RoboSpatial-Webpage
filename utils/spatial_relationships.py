# Utils for general spatial relationships

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import ConvexHull

from utils.spatial_relationship_utils import get_box2d_coordinates_and_depth, is_point_in_box, get_visible_points, calculate_visible_center_and_bounds, find_closest_vector_key, box_intersection, project_points_to_image, calculate_unoccupied_pixels, clip_bounding_box

DEBUG=False



def get_boxes_relationship(obj1, obj2, extrinsic, intrinsic, image_size):
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
    

    

    def determine_camera_and_world_relationship(box1_coords, box2_coords, box1_depth, box2_depth, box1_world_coords, box2_world_coords, image_size):
        box1_coords = np.array(box1_coords)
        box2_coords = np.array(box2_coords)
        box1_world_coords = np.array(box1_world_coords)
        box2_world_coords = np.array(box2_world_coords)
        box1_depth = np.array(box1_depth)
        box2_depth = np.array(box2_depth)

        # Calculate centers and bounds based on visible points within the image size
        box1_center_camera, box1_min_camera, box1_max_camera = calculate_visible_center_and_bounds(box1_coords, image_size)
        box2_center_camera, box2_min_camera, box2_max_camera = calculate_visible_center_and_bounds(box2_coords, image_size)

        # # Calculate bounding box min and max for 2D coords
        # box1_min, box1_max = np.min(box1_coords, axis=0), np.max(box1_coords, axis=0)
        # box2_min, box2_max = np.min(box2_coords, axis=0), np.max(box2_coords, axis=0)

        # Calculate min and max for depth
        box1_depth_min, box1_depth_max = np.min(box1_depth), np.max(box1_depth)
        box2_depth_min, box2_depth_max = np.min(box2_depth), np.max(box2_depth)

        # Calculate min and max for world coordinates (only Z axis)
        box1_world_z_min, box1_world_z_max = np.min(box1_world_coords[:, 2]), np.max(box1_world_coords[:, 2])
        box2_world_z_min, box2_world_z_max = np.min(box2_world_coords[:, 2]), np.max(box2_world_coords[:, 2])

        def get_relationship_strict():
            horizontal_relation = "overlapping"
            if np.all(box1_max_camera[0] < box2_min_camera[0]):
                horizontal_relation = "left"
            elif np.all(box1_min_camera[0] > box2_max_camera[0]):
                horizontal_relation = "right"

            depth_relation = "overlapping"
            if box1_depth_max < box2_depth_min:
                depth_relation = "in front of"
            elif box1_depth_min > box2_depth_max:
                depth_relation = "behind"

            vertical_world_relation = "overlapping"
            if box1_world_z_max < box2_world_z_min:
                vertical_world_relation = "below"
            elif box1_world_z_min > box2_world_z_max:
                vertical_world_relation = "above"

            return {
                "left": horizontal_relation == "left",
                "right": horizontal_relation == "right",
                "infront": depth_relation == "in front of",
                "behind": depth_relation == "behind",
                "cam_overlapping": horizontal_relation == "overlapping",
                "above": vertical_world_relation == "above",
                "below": vertical_world_relation == "below",
                "world_overlapping": vertical_world_relation == "overlapping"
            }

        def get_relationship_lenient():
            box1_in_box2 = is_point_in_box(box1_center_camera, box2_min_camera, box2_max_camera)
            box2_in_box1 = is_point_in_box(box2_center_camera, box1_min_camera, box1_max_camera)

            horizontal_relation = "overlapping"
            if box1_center_camera[0] < box2_center_camera[0] and not box2_in_box1:
                horizontal_relation = "left"
            elif box1_center_camera[0] > box2_center_camera[0] and not box1_in_box2:
                horizontal_relation = "right"
            
            # print()
            # print(box1_in_box2, box2_in_box1)
            # print(box1_min_camera, box1_max_camera, box2_min_camera, box2_max_camera)
            # print(box1_center_camera, box2_center_camera)
            # print()

            # depth_relation = "overlapping"
            # if box1_center[1] < box2_center[1]:
            #     depth_relation = "in front of"
            # elif box1_center[1] > box2_center[1]:
            #     depth_relation = "behind"
            
            #NOTE don't do lenient for in front of / behind
            depth_relation = "overlapping"
            if box1_depth_max < box2_depth_min:
                depth_relation = "in front of"
            elif box1_depth_min > box2_depth_max:
                depth_relation = "behind"
            
            vertical_world_relation = "overlapping"
            if box1_world_z_max < box2_world_z_min:
                vertical_world_relation = "below"
            elif box1_world_z_min > box2_world_z_max:
                vertical_world_relation = "above"

            # vertical_world_relation = "overlapping"
            # if box1_center[2] < box2_center[2]:
            #     vertical_world_relation = "below"
            # elif box1_center[2] > box2_center[2]:
            #     vertical_world_relation = "above"

            return {
                "left": horizontal_relation == "left",
                "right": horizontal_relation == "right",
                "infront": depth_relation == "in front of",
                "behind": depth_relation == "behind",
                "cam_overlapping": horizontal_relation == "overlapping",
                "above": vertical_world_relation == "above",
                "below": vertical_world_relation == "below",
                "world_overlapping": vertical_world_relation == "overlapping"
            }

        return {
            "strict": get_relationship_strict(),
            "lenient": get_relationship_lenient()
        }


    def determine_camera_and_world_relationship_strict(box1_coords, box2_coords, box1_depth, box2_depth, box1_world_coords, box2_world_coords, image_size):
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
        
        # def check_overlap(box1, box2, threshold_percentage=0.5):
        #     box1_points = np.asarray(box1.get_box_points())
        #     box2_points = np.asarray(box2.get_box_points())

        #     def project_points(points, axis):
        #         return np.dot(points, axis)

        #     def overlap_on_axis(box1_proj, box2_proj, threshold_percentage):
        #         box1_min, box1_max = np.min(box1_proj), np.max(box1_proj)
        #         box2_min, box2_max = np.min(box2_proj), np.max(box2_proj)

        #         box1_length = box1_max - box1_min
        #         box2_length = box2_max - box2_min
                
        #         overlap_amount = min(box1_max, box2_max) - max(box1_min, box2_min)
                
        #         box1_threshold = threshold_percentage * box1_length
        #         box2_threshold = threshold_percentage * box2_length

        #         print()
        #         print(overlap_amount)
        #         print(box1_length)
        #         print(box2_length)
        #         print()
                
        #         return overlap_amount >= box1_threshold and overlap_amount >= box2_threshold

        #     axes = np.vstack((np.diff(box1_points, axis=0), 
        #                     np.diff(box2_points, axis=0))).reshape(-1, 3)
        #     print(axes)
        #     for axis in axes:
        #         if not overlap_on_axis(project_points(box1_points, axis),
        #                             project_points(box2_points, axis),
        #                             threshold_percentage):
        #             return False, False, False

        #     return True, True, True
        

        # def check_overlap_complex(box1, box2, threshold_percentage=0.5):
        #     box1_points = np.asarray(box1.get_box_points())
        #     box2_points = np.asarray(box2.get_box_points())

        #     def project_points(points, axis):
        #         return np.dot(points, axis)

        #     def overlap_on_axis(box1_proj, box2_proj, threshold_percentage):
        #         box1_min, box1_max = np.min(box1_proj), np.max(box1_proj)
        #         box2_min, box2_max = np.min(box2_proj), np.max(box2_proj)

        #         box1_length = box1_max - box1_min
        #         box2_length = box2_max - box2_min
                
        #         overlap_amount = min(box1_max, box2_max) - max(box1_min, box2_min)
                
        #         box1_threshold = threshold_percentage * box1_length
        #         box2_threshold = threshold_percentage * box2_length
                
        #         return overlap_amount >= box1_threshold and overlap_amount >= box2_threshold

        #     # Function to get the local axes directions of a box
        #     def get_box_axes(points):
        #         return np.array([
        #             points[1] - points[0],  # x-axis direction
        #             points[3] - points[0],  # y-axis direction
        #             points[4] - points[0]   # z-axis direction
        #         ])

        #     # Get local axes for each box
        #     box1_axes = get_box_axes(box1_points)
        #     box2_axes = get_box_axes(box2_points)

        #     # Check overlap for x, y, z axes of both boxes
        #     x_overlap_box1 = overlap_on_axis(project_points(box1_points, box1_axes[0]),
        #                                     project_points(box2_points, box1_axes[0]),
        #                                     threshold_percentage)
        #     y_overlap_box1 = overlap_on_axis(project_points(box1_points, box1_axes[1]),
        #                                     project_points(box2_points, box1_axes[1]),
        #                                     threshold_percentage)
        #     z_overlap_box1 = overlap_on_axis(project_points(box1_points, box1_axes[2]),
        #                                     project_points(box2_points, box1_axes[2]),
        #                                     threshold_percentage)

        #     x_overlap_box2 = overlap_on_axis(project_points(box1_points, box2_axes[0]),
        #                                     project_points(box2_points, box2_axes[0]),
        #                                     threshold_percentage)
        #     y_overlap_box2 = overlap_on_axis(project_points(box1_points, box2_axes[1]),
        #                                     project_points(box2_points, box2_axes[1]),
        #                                     threshold_percentage)
        #     z_overlap_box2 = overlap_on_axis(project_points(box1_points, box2_axes[2]),
        #                                     project_points(box2_points, box2_axes[2]),
        #                                     threshold_percentage)

        #     return (x_overlap_box1 and x_overlap_box2,
        #             y_overlap_box1 and y_overlap_box2,
        #             z_overlap_box1 and z_overlap_box2)


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
        box1_coords, box2_coords, box1_depth, box2_depth, box1_world_coords, box2_world_coords, image_size)

    obj_centric_horizontal, obj_centric_depth = get_object_centric_relationship(obj1, obj2)

    relationships = {
        "camera_centric": {
            "left": cam_world_relations["lenient"]["left"],
            "right": cam_world_relations["lenient"]["right"],
            "infront": cam_world_relations["lenient"]["infront"],
            "behind": cam_world_relations["lenient"]["behind"],
            "above": cam_world_relations["lenient"]["above"],
            "below": cam_world_relations["lenient"]["below"],
            "overlapping": cam_world_relations["lenient"]["cam_overlapping"],
        },
        "world_centric": {
            "left": cam_world_relations["lenient"]["left"],
            "right": cam_world_relations["lenient"]["right"],
            "infront": cam_world_relations["lenient"]["infront"],
            "behind": cam_world_relations["lenient"]["behind"],
            "above": cam_world_relations["lenient"]["above"],
            "below": cam_world_relations["lenient"]["below"],
            "overlapping": cam_world_relations["lenient"]["world_overlapping"]
        },
        "object_centric": {
            "left": obj_centric_horizontal == "left",
            "right": obj_centric_horizontal == "right",
            "infront": obj_centric_depth == "in front of",
            "behind": obj_centric_depth == "behind",
            "above": cam_world_relations["lenient"]["above"],
            "below": cam_world_relations["lenient"]["below"],
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

    if type(floor_box) == list:
        min_bound = floor_box[0]
        max_bound = floor_box[1]
    else:
        min_bound = floor_box.get_min_bound()[:2]
        max_bound = floor_box.get_max_bound()[:2]
    
    x_range = np.arange(min_bound[0], max_bound[0], grid_resolution)
    y_range = np.arange(min_bound[1], max_bound[1], grid_resolution)
    
    return np.meshgrid(x_range, y_range)

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
def find_empty_areas(occupied):
    empty_areas = np.logical_not(occupied)
    return empty_areas

def get_empty_space(floor_box, boxes, grid_resolution=0.01):
    grid = create_floor_grid(floor_box, grid_resolution)
    empty_occupied = np.zeros(grid[0].shape, dtype=bool)
    if type(floor_box) != list:
        empty_occupied = mark_occupied_areas(grid, [floor_box], empty_occupied, floor=True)
    occupied = mark_occupied_areas(grid, boxes, empty_occupied)
    empty_areas = find_empty_areas(occupied)

    if DEBUG:
        # Plot the grid to visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(empty_areas, extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()], origin='lower', cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Empty Areas on the Floor')
        plt.show()

        # # Plot the grid to visualize
        # plt.figure(figsize=(10, 10))
        # plt.imshow(empty_areas, extent=[floor_box.get_min_bound()[0], floor_box.get_max_bound()[0], floor_box.get_min_bound()[1], floor_box.get_max_bound()[1]], origin='lower', cmap='gray')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Empty Areas on the Floor')
        # plt.show()
        # exit()

    return empty_areas, grid, occupied



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

        x_min, y_min = np.min(grid[0]), np.min(grid[1])
        x_max, y_max = np.max(grid[0]), np.max(grid[1])


        path_points_out_of_bounds = any([
            (point[0] < x_min or point[0] > x_max or point[1] < y_min or point[1] > y_max)
            for point in projected_points
        ])

        if path_points_out_of_bounds:
            return False

        ## Check each grid point if it is inside the hull
        for idx in np.ndindex(grid[0].shape):
            i, j = idx
            point = (grid[0][i, j], grid[1][i, j])
            if path.contains_point(point):
                if occupied[i, j]:
                    return False
        return True

    # Get the empty space on the floor
    empty_areas, grid, occupied = get_empty_space(floor_box, environment_boxes, 0.01)

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
                debug_empty_areas = find_empty_areas(debug_occupied)


                # Plot the grid to visualize
                plt.figure(figsize=(10, 10))
                plt.imshow(debug_empty_areas, extent=[debug_grid[0].min(), debug_grid[0].max(), debug_grid[1].min(), debug_grid[1].max()], origin='lower', cmap='gray')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'{frame} {direction} {results[frame][direction]}')
                plt.show()

    return results



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


def get_point_in_space_relative_to_object(floor_box, environment_boxes, box_a, extrinsic, intrinsic, image_size, have_face, num_samples, threshold=0.1, grid_resolution=0.1):
    """
    Returns uniformly sampled points in an empty area within a certain distance and direction to the object referred by the bounding box.

    Args:
        floor_box (o3d.geometry.OrientedBoundingBox): Floor bounding box.
        environment_boxes (list[o3d.geometry.OrientedBoundingBox]): List of bounding boxes for environment objects.
        box_a (o3d.geometry.OrientedBoundingBox): Object A's bounding box.
        extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world transformation.
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.
        threshold (float): Distance threshold to consider for proximity to the object.
        grid_resolution (float): Resolution of the grid for occupancy checking.
        num_samples (int): Number of points to uniformly sample.

    Returns:
        dict: Dictionary of lists of uniformly sampled points in the empty area within the specified distance to the object,
              for both object-centric and camera-centric frames.
    """
    # Get the empty space on the floor
    empty_areas, grid, occupied = get_empty_space(floor_box, environment_boxes, grid_resolution)

    # Project Box A to floor
    box_a_corners = project_to_floor(box_a)

    # Check the distance from each grid point to the bounding box
    def point_to_bbox_distance(point, bbox_corners):
        distances = np.linalg.norm(bbox_corners - point, axis=1)
        return np.min(distances)

    # List to store points within the threshold distance to the object
    points_within_distance = []

    for idx in np.ndindex(grid[0].shape):
        i, j = idx
        point = np.array([grid[0][i, j], grid[1][i, j]])
        if not occupied[i, j]:
            if point_to_bbox_distance(point, box_a_corners) <= threshold:
                points_within_distance.append((point[0], point[1]))

    def direction_check(point, box_corners, direction_vector):
        center = np.mean(box_corners, axis=0)
        vector_to_point = point - center
        direction_vector_2d = direction_vector[:2]  # Projecting direction vector to 2D
        dot_product = np.dot(vector_to_point, direction_vector_2d)
        return dot_product > 0

    def is_point_in_image(point, image_size):
        return 0 <= point[0] < image_size[1] and 0 <= point[1] < image_size[0]
    
    def is_point_occupied(occupancy_map, point):
        """
        Check if a 2D point is occupied in the occupancy map.

        Parameters:
        occupancy_map (np.array): The occupancy map generated by calculate_unoccupied_pixels.
        point (tuple): A tuple (x, y) representing the 2D point.

        Returns:
        bool: True if the point is occupied, False if it is unoccupied.
        """
        x, y = point

        # Ensure the point is within the bounds of the image
        if x < 0 or y < 0 or x >= occupancy_map.shape[1] or y >= occupancy_map.shape[0]:
            return True

        # Return whether the point is occupied or not
        return occupancy_map[y, x]

    # Object-centric directions
    object_directions = {
        'infront': np.asarray(box_a.R)[:, 0],  # Box A's facing direction
        'left': np.asarray(box_a.R)[:, 1],     # Perpendicular to the facing direction (left direction)
        'behind': -np.asarray(box_a.R)[:, 0],  # Opposite to Box A's facing direction
        'right': -np.asarray(box_a.R)[:, 1]    # Opposite to left direction
    }

    camera_infront = find_closest_vector_key(np.asarray(-1 * extrinsic[:, :3][:, 2][:3]), object_directions)
    camera_behind = find_closest_vector_key(np.asarray(extrinsic[:, :3][:, 2][:3]),  object_directions)
    camera_left = find_closest_vector_key(np.asarray(-1 * extrinsic[:3, 0]),  object_directions)
    camera_right = find_closest_vector_key(np.asarray(extrinsic[:3, 0]),  object_directions)

    # Camera-centric directions
    camera_directions = {
        'infront': object_directions[camera_infront],       # Camera's forward direction (z-axis)
        'behind': object_directions[camera_behind],           # Opposite to camera's forward direction
        'left': object_directions[camera_left],           # Camera's left direction (x-axis)
        'right': object_directions[camera_right]            # Opposite to camera's left direction
    }

    # # Camera-centric directions
    # camera_directions = {
    #     'infront': -np.asarray(extrinsic[:, :3][:, 2]),  # Camera's forward direction (z-axis)
    #     'left': -np.asarray(extrinsic[:3, 0]),           # Camera's left direction (x-axis)
    #     'behind': np.asarray(extrinsic[:, :3][:, 2]),    # Opposite to camera's forward direction
    #     'right': np.asarray(extrinsic[:3, 0])            # Opposite to camera's left direction
    # }

    sampled_points = {
        'objectcentric': {key: [] for key in object_directions.keys()},
        'cameracentric': {key: [] for key in camera_directions.keys()}
    }

    # Return None if there's no points that satisfy the constraint
    if len(points_within_distance) == 0: 
        return sampled_points, []

    # Convert points_within_distance to a numpy array for projection

    z_coordinate = np.min(np.asarray(box_a.get_box_points())[:, 2])  # Get the z coordinate of the bottom of box_a
    points_within_distance_np = np.array(points_within_distance)
    points_within_distance_3d = np.hstack((points_within_distance_np, np.full((points_within_distance_np.shape[0], 1), z_coordinate)))


    # Project points to image
    points_pixel, points_depth = project_points_to_image(points_within_distance_3d, extrinsic, intrinsic)
    
    print(points_pixel)
    points_pixel_visible = []
    for point in points_pixel:
        if is_point_in_image(point, image_size):
            points_pixel_visible.append(point)


    occupancy_map = calculate_unoccupied_pixels(environment_boxes, extrinsic, intrinsic, image_size)
    print(occupancy_map)
    print(occupancy_map.shape)

    # Filter points based on direction and visibility
    for idx, point in enumerate(points_within_distance):
        image_point = points_pixel[idx]
        print(image_point)
        if not is_point_occupied(occupancy_map, image_point):
            for direction, vector in object_directions.items():
                if direction_check(np.array(point), box_a_corners, vector):
                    sampled_points['objectcentric'][direction].append(image_point)
            if have_face:
                for direction, vector in camera_directions.items():
                    if direction_check(np.array(point), box_a_corners, vector):
                        sampled_points['cameracentric'][direction].append(image_point)

    # Uniformly sample the points
    def uniform_sample(points, num_samples):
        if len(points) > num_samples:
            return random.sample(points, num_samples)
        else:
            return points

    for frame in sampled_points.keys():
        for direction in sampled_points[frame].keys():
            sampled_points[frame][direction] = uniform_sample(sampled_points[frame][direction], num_samples)

    return sampled_points, points_pixel_visible



