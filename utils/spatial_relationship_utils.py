# Utils for spatial relationships
import numpy as np
from matplotlib.path import Path

from visualization.img_drawer import ImageDrawer


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

def project_points_to_image(points, extrinsic, intrinsic):
    extrinsic_w2c = np.linalg.inv(extrinsic)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_img = intrinsic @ extrinsic_w2c @ points.transpose()
    points_img = points_img.transpose()
    points_pixel = np.zeros((points_img.shape[0], 2))
    points_depth = np.zeros(points_img.shape[0])

    for i in range(points_img.shape[0]):
        points_pixel[i] = points_img[i][:2] / np.abs(points_img[i][2])
        points_depth[i] = points_img[i][2]

    return points_pixel.astype(int).tolist(), points_depth.tolist()

def is_point_in_2d_box(point, box_min, box_max):
    return np.all(box_min <= point[:2]) and np.all(point[:2] <= box_max)

def is_point_in_box(point, box_min, box_max):
    return np.all(box_min <= point) and np.all(point <= box_max)

def get_visible_points(coords, image_size):
    return [coord for coord in coords if 0 <= coord[0] < image_size[0] and 0 <= coord[1] < image_size[1]]

def calculate_visible_center_and_bounds(coords, image_size):
    visible_coords = get_visible_points(coords, image_size)
    if not visible_coords:
        visible_coords = coords  # If no points are visible, use all points
    center = np.mean(visible_coords, axis=0)
    min_coords = np.min(visible_coords, axis=0)
    max_coords = np.max(visible_coords, axis=0)
    return center, min_coords, max_coords

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

def box_intersection(box1, box2):
    box1_corners = np.asarray(box1.get_box_points())
    box2_corners = np.asarray(box2.get_box_points())
    
    box1_min = box1_corners.min(axis=0)
    box1_max = box1_corners.max(axis=0)
    box2_min = box2_corners.min(axis=0)
    box2_max = box2_corners.max(axis=0)
    
    return np.all(box1_max >= box2_min) and np.all(box2_max >= box1_min)


def generate_points_in_bounding_box(corners, num_points):
    """
    Generates a set number of points inside the 2D bounding box.

    Parameters:
        corners (list): List of 4 tuples representing the 4 corners of the bounding box.
        num_points (int): The number of points to generate.

    Returns:
        points (ndarray): num_points x 2 ndarray representing the points inside the bounding box.
    """
    # Ensure corners are in the correct shape
    assert len(corners) == 4 and all(len(corner) == 2 for corner in corners), "Corners should be a list of 4 tuples (each representing a 2D point)."
    
    # Convert list to ndarray for easier manipulation
    corners = np.array(corners)
    
    # Get the bounding box limits
    min_x, min_y = np.min(corners, axis=0)
    max_x, max_y = np.max(corners, axis=0)
    
    # Create a path object for the polygon
    path = Path(corners)
    
    # Generate points and ensure they are inside the polygon
    points = []
    while len(points) < num_points:
        # Generate random point
        point = np.random.uniform([min_x, min_y], [max_x, max_y], (1, 2))
        # Check if the point is inside the polygon
        if path.contains_point(point[0]):
            points.append(point[0])
    
    return np.array(points)



def order_bbox_coords(bbox_coords):
    """
    Orders the bounding box coordinates to be in the order:
    top-left, top-right, bottom-right, bottom-left.

    Parameters:
    - bbox_coords: list of tuples, each tuple containing (x, y) coordinates of the bounding box corners

    Returns:
    - ordered_coords: list of tuples, ordered coordinates
    """
    # Sort coordinates by y-value (top to bottom)
    sorted_by_y = sorted(bbox_coords, key=lambda x: x[1])

    # Extract the top two and bottom two points
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # Sort the top two points by x-value (left to right)
    top_two_sorted = sorted(top_two, key=lambda x: x[0])

    # Sort the bottom two points by x-value (left to right)
    bottom_two_sorted = sorted(bottom_two, key=lambda x: x[0])

    # Combine the points in the correct order
    ordered_coords = [top_two_sorted[0], top_two_sorted[1], bottom_two_sorted[1], bottom_two_sorted[0]]

    return ordered_coords

def calculate_unoccupied_pixels(bounding_boxes, extrinsic_matrix, intrinsic_matrix, image_size):
    """
    Calculate the occupancy map indicating which pixels are occupied by bounding boxes.

    Parameters:
    bounding_boxes (list of open3d.geometry.OrientedBoundingBox): List of oriented bounding boxes.
    extrinsic_matrix (np.array): Camera extrinsic matrix of shape (4, 4).
    intrinsic_matrix (np.array): Camera intrinsic matrix of shape (4, 4).
    image_size (tuple): The size of the image as (width, height).

    Returns:
    np.array: An array where True indicates an unoccupied pixel and False indicates an occupied pixel.
    """
    width, height = image_size
    # Create an array to store the occupancy of each pixel
    occupancy_map = np.zeros((height, width), dtype=bool)

    # Loop through each bounding box
    for bbox in bounding_boxes:
        # Get the eight corners of the bounding box
        bbox_corners = np.asarray(bbox.get_box_points())

        # Transform the corners into the camera coordinate system using the extrinsic matrix
        bbox_corners_camera = (extrinsic_matrix @ np.hstack((bbox_corners, np.ones((8, 1)))).T).T[:, :4]

        # Project the corners onto the image plane using the intrinsic matrix
        bbox_corners_image = (intrinsic_matrix @ bbox_corners_camera.T).T
        bbox_corners_image = bbox_corners_image[:, :2] / bbox_corners_image[:, 3:4]

        # Determine the bounding box in the image plane
        min_x, min_y = np.min(bbox_corners_image, axis=0).astype(int)
        max_x, max_y = np.max(bbox_corners_image, axis=0).astype(int)

        # Clamp the coordinates to the image size
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width - 1, max_x)
        max_y = min(height - 1, max_y)

        # Mark the pixels within the bounding box as occupied
        occupancy_map[min_y:max_y+1, min_x:max_x+1] = True

    # Return the occupancy map, where False means unoccupied and True means occupied
    return occupancy_map

def clip_point(point, width, height):
    x, y = point
    x = np.clip(x, 0, width)
    y = np.clip(y, 0, height)
    return (x, y)

def clip_bounding_box(bbox, image_size):
    width, height = image_size
    clipped_bbox = np.asarray([clip_point(point, width, height) for point in bbox])
    return clipped_bbox


def convert_to_square_bounding_box(points):
    """
    Converts 8 corners of a bounding box into a square bounding box with 4 corners.

    Parameters:
        points (list of tuples or ndarray): A list or array of 8 points (x, y) representing the corners of the bounding box.

    Returns:
        ndarray: A 4x2 array containing the 4 corners of the square bounding box.
    """
    # Convert points to a numpy array if it's not already
    points = np.array(points)

    # Step 1: Find the minimum and maximum x and y coordinates
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # Step 2: Create the square bounding box
    square_corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

    return square_corners

def get_box3d_coordinates(box, extrinsic, intrinsic, image_size):
    """Get 3D box coordinates in the image plane.

    Args:
        box (open3d.geometry.OrientedBoundingBox): Box to be drawn.
        extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world
            transformation.
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.

    Returns:
        np.ndarray: Array of shape (8, 2) with 2D coordinates of the 3D box corners in the image plane.
    """
    extrinsic_w2c = np.linalg.inv(extrinsic)
    w, h = image_size

    camera_pos_in_world = (
        extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
    if ImageDrawer._inside_box(box, camera_pos_in_world):
        return None

    corners = np.asarray(box.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
    corners = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1)
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = corners_img.transpose()

    corners_pixel = np.zeros((corners_img.shape[0], 2))

    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])
 
    return corners_pixel

# Function to check if an oriented bounding box is on the floor
def is_bounding_box_on_floor(obb, floor_z=0.0, tolerance=0.4):
    # Extract the corner points of the bounding box
    corner_points = np.asarray(obb.get_box_points())
    
    # Get the minimum Z-coordinate (or Y if your floor plane is on that axis)
    min_z = np.min(corner_points[:, 2])  # Assuming floor is in the Z = 0 plane
    
    # Check if the bottom face is within the tolerance of the floor
    return np.abs(min_z - floor_z) <= tolerance