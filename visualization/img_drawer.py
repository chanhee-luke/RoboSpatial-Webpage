import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import path


class ImageDrawer:
    """Visualization tool for ego-centric images.

    This class serves as the API for visualizing ego-centric images.

    Args:
        image (str): Path of image to be visualized.
        verbose (bool): Whether to print related messages. Defaults to False.
    """

    def __init__(self, image, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('Loading image', image)
        img = cv2.imread(image)
        if self.verbose:
            print('Loading image Complete')
        img = img[:, :, ::-1].astype(np.float32)  # BGR to RGB
        self.occupied = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        self.img = img
        self.EPS = 1e-4
        self.ALPHA = 0.75

    def draw_text(self,
                  text,
                  font=cv2.FONT_HERSHEY_SIMPLEX,
                  pos=(0, 0),
                  size=(0, 0),
                  font_scale=1,
                  font_thickness=2,
                  text_color=(0, 255, 0),
                  text_color_bg=(0, 0, 0)):
        """Draw the category text on the image.

        Args:
            text (str): Text to be drawn.
            font (cv2.FONT_TYPE): The font type to be used to plot.
                Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            pos (tuple): The drawing position. Defaults to (0, 0).
            size (tuple): Image size. Defaults to (0, 0).
            font_scale (int): Font scale. Defaults to 1.
            font_thickness (int): Font thickness. Defaults to 2.
            text_color (tuple): Font color. Defaults to (0, 255, 0).
            text_color_bg (tuple): Background color for drawing texts.
                Defaults to (0, 0, 0).
        """

        x, y = pos
        w, h = size
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        if y * 2 > h:
            dy = -10
        else:
            dy = 10

        try:
            while self.occupied[y, x] or self.occupied[
                    y, x +
                    text_w] or self.occupied[y + text_h,
                                             x] or self.occupied[y + text_h,
                                                                 x + text_w]:
                y += dy
        except:  # noqa: E722
            pass
            # TODO
        cv2.rectangle(self.img, (x, y), (x + text_w, y + text_h),
                      text_color_bg, -1)
        cv2.putText(self.img, text, (x, y + text_h + font_scale - 1), font,
                    font_scale, text_color, font_thickness)

        self.occupied[y:y + text_h, x:x + text_w] = True

    def orig_draw_box3d(self, box, color, label, extrinsic, intrinsic):
        """Draw 3D boxes on the image.

        Args:
            box (open3d.geometry.OrientedBoundingBox): Box to be drawn.
            color (tuple): Box color.
            label (str): Box category label.
            extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world
                transformation.
            intrinsic (np.ndarray): 4x4 camera intrinsic matrix.
        """
        extrinsic_w2c = np.linalg.inv(extrinsic)
        h, w, _ = self.img.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((x, y)).T

        camera_pos_in_world = (
            extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
        if self._inside_box(box, camera_pos_in_world):
            return

        corners = np.asarray(box.get_box_points())
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
            if (corners_img[line][:, 2] < self.EPS).any():
                continue
            px = corners_pixel[line[0]].astype(np.int32)
            py = corners_pixel[line[1]].astype(np.int32)
            cv2.line(self.img, (px[0], px[1]), (py[0], py[1]), color, 2)

        all_mask = np.zeros((h, w), dtype=bool)
        for face in faces:
            if (corners_img[face][:, 2] < self.EPS).any():
                continue
            pts = corners_pixel[face]
            p = path.Path(pts[:, :2])
            mask = p.contains_points(pixel_points).reshape((h, w))
            all_mask = np.logical_or(all_mask, mask)
        self.img[all_mask] = self.img[all_mask] * self.ALPHA + (
            1 - self.ALPHA) * np.array(color)

        if (all_mask.any()):
            textpos = np.min(corners_pixel, axis=0).astype(np.int32)
            textpos[0] = np.clip(textpos[0], a_min=0, a_max=w)
            textpos[1] = np.clip(textpos[1], a_min=0, a_max=h)
            self.draw_text(label,
                           pos=textpos,
                           size=(w, h),
                           text_color=(255, 255, 255),
                           text_color_bg=color)






    def draw_box3d(self, box, color, label, extrinsic, intrinsic):
        """Draw 3D boxes on the image.

        Args:
            box (open3d.geometry.OrientedBoundingBox): Box to be drawn.
            color (tuple): Box color.
            label (str): Box category label.
            extrinsic (np.ndarray): 4x4 extrinsic matrix, camera to world
                transformation.
            intrinsic (np.ndarray): 4x4 camera intrinsic matrix.
        """
        extrinsic_w2c = np.linalg.inv(extrinsic)
        h, w, _ = self.img.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((x, y)).T

        camera_pos_in_world = (
            extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
        if self._inside_box(box, camera_pos_in_world):
            return

        corners = np.asarray(box.get_box_points())
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
            if (corners_img[line][:, 2] < self.EPS).any():
                continue
            px = corners_pixel[line[0]].astype(np.int32)
            py = corners_pixel[line[1]].astype(np.int32)
            cv2.line(self.img, (px[0], px[1]), (py[0], py[1]), color, 2)

        all_mask = np.zeros((h, w), dtype=bool)
        for face in faces:
            if (corners_img[face][:, 2] < self.EPS).any():
                continue
            pts = corners_pixel[face]
            p = path.Path(pts[:, :2])
            mask = p.contains_points(pixel_points).reshape((h, w))
            all_mask = np.logical_or(all_mask, mask)
        self.img[all_mask] = self.img[all_mask] * self.ALPHA + (
            1 - self.ALPHA) * np.array(color)

        if (all_mask.any()):
            textpos = np.min(corners_pixel, axis=0).astype(np.int32)
            textpos[0] = np.clip(textpos[0], a_min=0, a_max=w)
            textpos[1] = np.clip(textpos[1], a_min=0, a_max=h)
            self.draw_text(label,
                        pos=textpos,
                        size=(w, h),
                        text_color=(255, 255, 255),
                        text_color_bg=color)

        # Draw orientation arrow
        # Calculate the center of the front face (first four corners)
        front_face_center = np.mean(corners_pixel[:4], axis=0).astype(np.int32)
        # The direction vector (assuming the front face is oriented along the x-axis in the box's local frame)
        direction_vector = (corners_pixel[1] - corners_pixel[0]).astype(np.float32)

        # Normalize the direction vector and scale it to a fixed length
        fixed_length = 50
        if np.linalg.norm(direction_vector) > 0:
            direction_vector = direction_vector / np.linalg.norm(direction_vector) * fixed_length

        arrow_end = front_face_center + direction_vector

        # Check if the arrow start or end is out of bounds
        if (front_face_center[0] < 0 or front_face_center[0] >= w or
            front_face_center[1] < 0 or front_face_center[1] >= h or
            arrow_end[0] < 0 or arrow_end[0] >= w or
            arrow_end[1] < 0 or arrow_end[1] >= h):
            # Recalculate the arrow end to keep the arrow within the image bounds
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            max_length = min(
                (w - 1 - front_face_center[0]) / direction_vector[0] if direction_vector[0] > 0 else front_face_center[0] / -direction_vector[0],
                (h - 1 - front_face_center[1]) / direction_vector[1] if direction_vector[1] > 0 else front_face_center[1] / -direction_vector[1])
            arrow_end = front_face_center + direction_vector * min(fixed_length, max_length)

        # front_face_center[0] = np.clip(front_face_center[0], 0, w - 10)
        # front_face_center[1] = np.clip(front_face_center[1], 0, h - 10)
        # arrow_end[0] = np.clip(arrow_end[0], 0, w - 10)
        # arrow_end[1] = np.clip(arrow_end[1], 0, h - 10)

        arrow_end = arrow_end.astype(np.int32)

        cv2.arrowedLine(self.img, (front_face_center[0], front_face_center[1]), (arrow_end[0], arrow_end[1]), color, 2, tipLength=0.5)





    def show(self):
        """Show the image on the screen."""
        plt.imshow(self.img / 255.0)
        plt.show()

    @staticmethod
    def _inside_box(box, point):
        """Check if any points are in the box.

        Args:
            box (open3d.geometry.OrientedBoundingBox): Oriented Box.
            point (np.ndarray): N points represented by nx4 array (x, y, z, 1).

        Returns:
            bool: The result.
        """
        point_vec = o3d.utility.Vector3dVector(point[:, :3])
        inside_idx = box.get_point_indices_within_bounding_box(point_vec)
        if len(inside_idx) > 0:
            return True
        return False
