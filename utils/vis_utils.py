
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import numpy as np

from utils.spatial_relationship_utils import order_bbox_coords



def show_image_with_points_and_bbox_coords(image_path, points, bbox_coords):
    """
    Displays an image with 2D points and a bounding box overlay.

    Parameters:
    - image_path: str, path to the image file
    - points: list of tuples, each tuple containing (x, y) coordinates of the points
    - bbox_coords: list of tuples, each tuple containing (x, y) coordinates of the bounding box corners
                   The coordinates should be in the order: top-left, top-right, bottom-right, bottom-left

    Example:
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    bbox_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """    
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Overlay the points
    for point in points:
        ax.plot(point[0], point[1], 'ro')  # 'ro' stands for red color and circle marker
    
    # Create a Polygon patch for the bounding box
    polygon = patches.Polygon(bbox_coords, closed=True, linewidth=2, edgecolor='b', facecolor='none')
    
    # Add the patch to the Axes
    ax.add_patch(polygon)
    
    # Show the plot
    plt.show()

def show_image_with_bbox(image_path, bbox, extrinsic, intrinsic):
    """
    Display an image with an overlaid bounding box.

    :param image_path: Path to the image file.
    :param bbox: A list of four tuples, each containing the (x, y) coordinates of the bounding box vertices.
                 The order should be [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """

    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    extrinsic_w2c = np.linalg.inv(extrinsic)
    h, w, _ = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    camera_pos_in_world = (
        extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()

    corners = np.asarray(bbox.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
    corners = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1)
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = corners_img.transpose()
    corners_pixel = np.zeros((corners_img.shape[0], 2))

    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
            [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7],
            [0, 3, 7, 4], [1, 2, 6, 5]]
    for line in lines:
        if (corners_img[line][:, 2] < 1e-4).any():
            continue
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(image, (px[0], px[1]), (py[0], py[1]), [255, 255, 0], 2)
    
    # Show the plot
    plt.imshow(image / 255.0)
    plt.show()

def show_image_with_points(image_path, points):
    """
    Displays an image with 2D points overlay.

    Parameters:
    - image_path: str, path to the image file
    - points: list of tuples, each tuple containing (x, y) coordinates of the points

    Example:
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Overlay the points
    for point in points:
        ax.plot(point[0], point[1], 'ro')  # 'ro' stands for red color and circle marker
    
    # Show the plot
    plt.show()


def show_image_with_bbox_points(image_path, corners_pixel):
    """
    Displays an image with 2D points overlay.

    Parameters:
    - image_path: str, path to the image file
    - points: np.array of tuples, each tuple containing (x, y) coordinates of the points

    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Overlay the points
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
            [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for line in lines:
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(image, (px[0], px[1]), (py[0], py[1]), [255, 255, 0], 2)
    
    # Show the plot
    plt.imshow(image / 255.0)
    plt.show()


def show_image_with_square_bbox_points(image_path, corners_pixel):
    """
    Displays an image with 4-corner bounding box overlay.

    Parameters:
    - image_path: str, path to the image file
    - corners_pixel: np.array of tuples, each tuple containing (x, y) coordinates of the 4-corner bounding box

    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Overlay the bounding box
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for line in lines:
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(image, (px[0], px[1]), (py[0], py[1]), [255, 255, 0], 2)
    
    # Show the plot
    plt.imshow(image / 255.0)
    plt.show()


def show_image_with_points_and_bbox(image_path, bbox, points, extrinsic, intrinsic):
    """
    Display an image with an overlaid bounding box.

    :param image_path: Path to the image file.
    :param bbox: A list of four tuples, each containing the (x, y) coordinates of the bounding box vertices.
                 The order should be [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """

    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    extrinsic_w2c = np.linalg.inv(extrinsic)
    h, w, _ = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    camera_pos_in_world = (
        extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()

    corners = np.asarray(bbox.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
    corners = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1)
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = corners_img.transpose()
    corners_pixel = np.zeros((corners_img.shape[0], 2))

    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
            [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7],
            [0, 3, 7, 4], [1, 2, 6, 5]]
    for line in lines:
        if (corners_img[line][:, 2] < 1e-4).any():
            continue
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(image, (px[0], px[1]), (py[0], py[1]), [255, 255, 0], 2)
    
    # Overlay points on the image
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=15, color=(255, 0, 0), thickness=-1)
    
    # Show the plot
    plt.imshow(image / 255.0)
    plt.show()