import os
import pickle
from typing import List, Union
import json

import numpy as np


class MultiScanLoader:
    def __init__(self,
                 data_root: dict,
                 verbose: bool = False,
                 color_setting: str = None,
                 thickness: float = 0.01):
        
        self.verbose = verbose
        self.data_root = data_root

        # Load image list
        self.image_stats = dict()
        self.data_list = dict()

        if self.verbose:
            print("Loading multiscan...")
            print('Multiscan Dataset root')
            print("multiscan", ':', data_root)

        num_scenes = 0
        num_images = 0
        with open(os.path.join(data_root["image_path"], "stats.jsonl"), 'r') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                for key, values in json_obj.items():
                    self.data_list[key] = values
                    num_images += len(values)
                    num_scenes += 1

        self.image_stats = {
            "num_scenes" :  num_scenes,
            "num_images" :  num_images
        }

        if self.verbose:
            print(f"Loaded {num_images} images from {num_scenes} scenes from multiscan")
    
    def list_data(self):
        return self.data_list.items()

    def list_scenes(self):
        for scene_id in self.data_list.keys():
            yield "multiscan", scene_id

    def list_ann(self, scene_id):
        ann = dict()
        # Load annotation files
        ## OBB annotations
        with open(os.path.join(self.data_root["dataset_path"], scene_id, f"{scene_id}.annotations.json"), "r") as file:
            ann["obb_annotations"] = json.load(file)

        ## Camera intrinstics for the scene
        camera_poses = {}
        idx = 0
        with open(os.path.join(self.data_root["dataset_path"], scene_id, f"{scene_id}.jsonl"), "r") as file:
            for line in file:
                
                if idx not in self.data_list[scene_id]:
                    idx += 1
                    continue

                camera_pose = json.loads(line)

                ## Format camera poses to match open3d
                # Convert transform to 4x4 matrix
                transform = np.array(camera_pose['transform']).reshape((4, 4), order='F')

                # Convert ARKit camera coordinates to Open3D camera coordinates
                transform = np.dot(transform, np.diag([1, -1, -1, 1]))
                transform /= transform[3, 3]

                from scipy.spatial.transform import Rotation as R
                translation = transform[:3, 3]

                # Using Euler Angles
                euler_angles = np.asarray(camera_pose.get('euler_angles'))
                rotation_matrix_euler = R.from_euler('xyz', euler_angles).as_matrix()

                # Using Quaternion
                quaternion = np.asarray(camera_pose.get('quaternion'))
                quaternion_xyzw = quaternion[[1, 2, 3, 0]]
                rotation_matrix_quat = R.from_quat(quaternion_xyzw).as_matrix()

                # Combine rotation matrix and translation vector to form the extrinsic matrix (row-major)
                extrinsic_matrix_euler = np.eye(4)
                extrinsic_matrix_euler[:3, :3] = rotation_matrix_euler
                extrinsic_matrix_euler[:3, 3] = translation

                extrinsic_matrix_quat = np.eye(4)
                extrinsic_matrix_quat[:3, :3] = rotation_matrix_quat
                extrinsic_matrix_quat[:3, 3] = translation

                # Convert the column-major extrinsic matrix to row-major by transposing the rotation part
                extrinsic_matrix_euler[:3, :3] = extrinsic_matrix_euler[:3, :3].T
                extrinsic_matrix_quat[:3, :3] = extrinsic_matrix_quat[:3, :3].T


                # Inverse the transformation matrix to get camera extrinsics
                # camera_extrinsics = np.linalg.inv(transform) # Don't do this to match embodiedscan
                camera_extrinsics = extrinsic_matrix_quat

                

                # Convert intrinsics to 4x4 matrix
                camera_intrinsics = np.eye(4)
                camera_intrinsics[:3, :3] = np.array(camera_pose['intrinsics']).reshape((3, 3), order='F')

                # Convert intrinsics to 4x4 column-major matrix
                camera_intrinsics_col_major = np.eye(4)
                camera_intrinsics_col_major[:3, :3] = np.array(camera_pose['intrinsics']).reshape((3, 3), order='F')

                # Now, convert the column-major matrix to row-major matrix
                # Step 1: Extract the 3x3 part of the matrix
                intrinsics_3x3_col_major = camera_intrinsics_col_major[:3, :3]

                # Step 2: Reshape it to row-major
                intrinsics_3x3_row_major = intrinsics_3x3_col_major.T

                # Step 3: Create a new 4x4 row-major matrix and insert the 3x3 part
                camera_intrinsics_row_major = np.eye(4)
                camera_intrinsics_row_major[:3, :3] = intrinsics_3x3_row_major


                camera_pose["cam2global"] = transform
                camera_pose["cam2img"] = camera_intrinsics_col_major

                # print(extrinsic_matrix_quat)
                # print(extrinsic_matrix_euler)
                # print(transform)
                # print()
                # print(camera_pose["transform"])
                # print(camera_pose["quaternion"])
                # print(camera_pose["euler_angles"])
                # print(camera_intrinsics)
                # print(camera_intrinsics_row_major)
                # print(camera_intrinsics_col_major)
                # exit()

                camera_poses[idx] = camera_pose
                idx += 1

            ann["camera_poses"] = camera_poses
        
        return ann
    
    def list_images_and_instances(self, scene_id):
        ann = {}
        
        ann = self.list_ann(scene_id)
        
        # for obj in ann["obb_annotations"]["objects"]:
        #     print(obj)
        #     print(ann["camera_intrinsics"].keys())
        #     exit()

        # print(ann["camera_poses"])
        # print(self.data_list[scene_id])
        # exit()
        
        
        return ann["obb_annotations"]["objects"], ann["camera_poses"]

        # res.append({
        #             'bbox_3d': instance['bbox_3d'], # 9DOF Box annotation
        #             'bbox_id': instance['bbox_id'], # id for the box (for the specific dataset), starts from 1
        #             'name': label, # Name of the object
        #             'bbox_label_3d': instance['bbox_label_3d'], # Category label (across datasets),
        #             'axis_align_matrix': sample["axis_align_matrix"], # Same for one scene
        #             'cam2img': cam2img, # Same for one scene
        #             'depth_cam2img': depth_cam2img
        #         })