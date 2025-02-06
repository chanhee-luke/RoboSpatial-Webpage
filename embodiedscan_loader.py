import os
import pickle
from typing import List, Union
from collections import defaultdict
import json

import numpy as np
import open3d as o3d

from visualization.color_selector import ColorMap
from visualization.continuous_drawer import (
    ContinuousDrawer, ContinuousOccupancyDrawer)
from visualization.img_drawer import ImageDrawer
from visualization.utils import _9dof_to_box, _box_add_thickness

DATASETS = ['scannet', '3rscan', 'matterport3d']

class EmbodiedScanLoader:
    """Scan loader, based from EmbodiedScan Explorer.

    This class serves as the API for analyze and visualize EmbodiedScan
    dataset with demo data.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        verbose (bool): Whether to print related messages. Defaults to False.
        color_setting (str, optional): Color settings for visualization.
            Defaults to None.
            Accept the path to the setting file like
                embodiedscan/visualization/full_color_map.txt
        thickness (float): Thickness of of the displayed box lines.
    """

    def __init__(self,
                 data_root: dict,
                 ann_file: list,
                 verbose: bool = False,
                 color_setting: str = None,
                 thickness: float = 0.01):

        self.ann_files = ann_file
        self.data_root = data_root
        self.verbose = verbose
        self.thickness = thickness

        if self.verbose:
            print('Embodiedscan Dataset roots')
            for dataset, path in data_root.items():
                print(dataset, ':', path)

        if self.verbose:
            print('Loading EmbodiedScan...')
        self.metainfo = None
        ## Load embodiedscan annotated scan datasets (scannet, matterport3d, 3rscan)
        data_list = []
        for file in self.ann_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            if self.metainfo is None:
                self.metainfo = data['metainfo']
            else:
                assert self.metainfo == data['metainfo']
            
            data_list += data['data_list']


        if isinstance(self.metainfo['categories'], list):
            self.classes = self.metainfo['categories']
            self.id_to_index = {i: i for i in range(len(self.classes))}
        elif isinstance(self.metainfo['categories'], dict):
            self.classes = list(self.metainfo['categories'].keys())
            self.id_to_index = {
                i: self.classes.index(classes)
                for classes, i in self.metainfo['categories'].items()
            }
        self.color_selector = ColorMap(classes=self.classes,
                                       init_file=color_setting)
        
        # Check if certain scan exists
        self.data = defaultdict(list)
        for data in data_list:
            splits = data['sample_idx'].split('/')
            dataset = splits[0]
            if dataset not in self.data_root.keys(): # Skip dataset that is not loaded
                continue

            data['dataset'] = dataset
            if self.data_root[dataset] is not None:
                if dataset == 'scannet':
                    region = splits[1]
                    dirpath = os.path.join(self.data_root['scannet'], 'scans',
                                           region)
                elif dataset == '3rscan':
                    region = splits[1]
                    dirpath = os.path.join(self.data_root['3rscan'], region)
                elif dataset == 'matterport3d':
                    building, region = splits[1], splits[2]
                    dirpath = os.path.join(self.data_root['matterport3d'],
                                           building)
                else:
                    region = splits[1]
                    dirpath = os.path.join(self.data_root[dataset], region)
                if os.path.exists(dirpath):
                    self.data[dataset].append(data)
        
        # self.dataset_stats = {}
        # for dataset, data in self.data.items():
        #     self.dataset_stats[dataset] = len(data)

        if self.verbose:
            for dataset, data in self.data.items():
                print(f"Loaded {len(data)} scenes from {dataset}")
            print('Loading complete')

    def count_scenes(self, datasets:list):
        """Count the number of scenes."""
        total_scenes = 0
        for dataset, data in self.data.items():
            if dataset not in datasets:
                continue
            total_scenes += len(data)
        return total_scenes

    def list_categories(self):
        """List the categories involved in the dataset."""
        res = []
        for cate, id in self.metainfo['categories'].items():
            res.append({'category': cate, 'id': id})
        return res

    # def list_scenes(self, datasets:list):
    #     """List all scenes in the dataset."""
    #     res = []
    #     for dataset, data in self.data.items():
    #         if dataset not in datasets:
    #             continue
    #         for scene in data:
    #             res.append((dataset, scene['sample_idx']))
    #     return res

    def list_scenes(self, datasets:list):
        """List all scenes in the dataset."""
        res = []
        for dataset, data in self.data.items():
            if dataset not in datasets:
                continue
            
            for i, scene in enumerate(data):
                yield dataset, i, scene['sample_idx']

    def list_cameras(self, dataset_name, scene_name):
        """List all the camera frames in the scene.

        Args:
            scene_name (str): Scene name.

        Returns:
            list[str] or None: List of all the frame names. If there is no
            frames, we will return None.
        """
        for sample in self.data[dataset_name]:
            if sample['sample_idx'] == scene_name:
                res = []
                dataset = sample['dataset']
                for img in sample['images']:
                    img_path = img['img_path']
                    if dataset == 'scannet':
                        cam_name = img_path.split('/')[-1][:-4]
                    elif dataset == '3rscan':
                        cam_name = img_path.split('/')[-1][:-10]
                    elif dataset == 'matterport3d':
                        cam_name = img_path.split(
                            '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                    else:
                        cam_name = img_path.split('/')[-1][:-4]
                    res.append(cam_name)
                return res

        print(f'No such scene {scene_name} in {dataset_name}')
        return None

    def list_images(self, dataset_name, scene_name):
        """List all the camera frames in the scene.

        Args:
            scene_name (str): Scene name.

        Returns:
            list[str] or None: List of all the frame names. If there is no
            frames, we will return None.
        """
        images = []

        for sample in self.data[dataset_name]:
            if sample['sample_idx'] == scene_name:
                images = sample['images']
        if len(images) == 0:
            print(f'No such scene {scene_name} in {dataset_name}')
            return None
        else:
            return images

    def list_instances(self, dataset_name, scene_name):
        """List all the instance annotations in the scene.

        Args:
            scene_name (str): Scene name.

        Returns:
            list[dict] or None: List of all the instance annotations. If there
            is no instances, we will return None.
        """
        for sample in self.data[dataset_name]:
            if sample['sample_idx'] == scene_name:
                res = []
                for instance in sample['instances']:
                    if "cam2img" not in sample: #NOTE for matterport3d
                        cam2img = sample['images'][0]['cam2img']
                        depth_cam2img = []
                    else:
                        cam2img = sample['cam2img'] if 'cam2img' in sample else []
                        depth_cam2img = sample['depth_cam2img'] if 'depth_cam2img' in sample else []
                    label = self.classes[self.id_to_index[
                        instance['bbox_label_3d']]]
                    res.append({
                        'bbox_3d': instance['bbox_3d'], # 9DOF Box annotation
                        'bbox_id': instance['bbox_id'], # id for the box (for the specific dataset), starts from 1
                        'name': label, # Name of the object
                        'bbox_label_3d': instance['bbox_label_3d'], # Category label (across datasets),
                        'axis_align_matrix': sample["axis_align_matrix"], # Same for one scene
                        'cam2img': cam2img, # Same for one scene
                        'depth_cam2img': depth_cam2img
                    })
                return res

        print('No such scene')
        return None

    def list_image_anns(self, dataset_name, scene_name, image_name):
        """List all the instance annotations in the image.

        Args:
            scene_name (str): Scene name.
            image_name (str): Image name.

        Returns:
            list[dict] or None: List of all the instance annotations. If there
            is no instances, we will return None.
        """
        anns = None
        for sample in self.data[dataset_name]:
            if sample['sample_idx'] == scene_name:
                anns = {}
                dataset = sample['dataset']
                for img in sample['images']:
                    img_path = img['img_path']
                    if dataset == 'scannet':
                        cam_name = img_path.split('/')[-1][:-4]
                    elif dataset == '3rscan':
                        cam_name = img_path.split('/')[-1][:-10]
                    elif dataset == 'matterport3d':
                        cam_name = img_path.split(
                            '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                    else:
                        cam_name = img_path.split('/')[-1][:-4]
                    if cam_name == image_name:
                        anns = sample['images']
                        anns["axis_aligned_matrix"] = sample["axis_aligned_matrix"]
                        break
        if not anns:
            print("No annotation for scene {scene_name} and image {image_name} in {dataset_name}")
        return anns

    def scene_info(self, dataset_name, scene_name):
        """Show the info of the given scene.

        Args:
            scene_name (str): Scene name.

        Returns:
            dict or None: Dict of scene info. If there is no such a scene, we
            will return None.
        """
        for scene in self.data[dataset_name]:
            if scene['sample_idx'] == scene_name:
                if self.verbose:
                    print('Info of', scene_name)
                    print(len(scene['images']), 'images')
                    print(len(scene['instances']), 'boxes')
                return dict(num_images=len(scene['images']),
                            num_boxes=len(scene['instances']))

        if self.verbose:
            print('No such scene')
        return None

    def render_scene(self, scene_name, render_box=False):
        """Render a given scene with open3d.

        Args:
            scene_name (str): Scene name.
            render_box (bool): Whether to render the box in the scene.
                Defaults to False.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        select = None
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                select = scene
                break
        axis_align_matrix = select['axis_align_matrix']
        if dataset == 'scannet':
            filepath = os.path.join(self.data_root['scannet'], 'scans', region,
                                    f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            filepath = os.path.join(self.data_root['3rscan'], region,
                                    'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.data_root['matterport3d'], building,
                                    'region_segmentations', f'{region}.ply')
        else:
            raise NotImplementedError

        if self.verbose:
            print('Loading mesh')
        mesh = o3d.io.read_triangle_mesh(filepath, True)
        mesh.transform(axis_align_matrix)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if self.verbose:
            print('Loading complete')
        boxes = []
        if render_box:
            if self.verbose:
                print('Rendering box')
            for instance in select['instances']:
                box = _9dof_to_box(
                    instance['bbox_3d'],
                    self.classes[self.id_to_index[instance['bbox_label_3d']]],
                    self.color_selector)
                boxes += _box_add_thickness(box, self.thickness)
            if self.verbose:
                print('Rendering complete')
        o3d.visualization.draw_geometries([mesh, frame] + boxes)

    def render_continuous_scene(self,
                                scene_name,
                                start_cam=None,
                                pcd_downsample=100):
        """Render a scene with continuous ego-centric observations.

        Args:
            scene_name (str): Scene name.
            start_cam (str, optional): Camera frame from which the rendering
                starts. Defaults to None, corresponding to the first frame.
            pcd_downsample (int): The downsampling ratio of point clouds.
                Defaults to 100.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        selected_scene = None
        start_idx = -1
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                selected_scene = scene
                if start_cam is not None:
                    start_idx = -1
                    for i, img in enumerate(scene['images']):
                        img_path = img['img_path']
                        if dataset == 'scannet':
                            cam_name = img_path.split('/')[-1][:-4]
                        elif dataset == '3rscan':
                            cam_name = img_path.split('/')[-1][:-10]
                        elif dataset == 'matterport3d':
                            cam_name = img_path.split(
                                '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                        else:
                            cam_name = img_path.split('/')[-1][:-4]
                        if cam_name == start_cam:
                            start_idx = i
                            break
                    if start_idx == -1:
                        print('No such camera')
                        return
                else:
                    start_idx = 0

        if selected_scene is None:
            print('No such scene')
            return

        drawer = ContinuousDrawer(dataset, self.data_root[dataset],
                                  selected_scene, self.classes,
                                  self.id_to_index, self.color_selector,
                                  start_idx, pcd_downsample, self.thickness)
        drawer.begin()

    def render_continuous_occupancy(self, scene_name, start_cam=None):
        """Render occupancy with continuous ego-centric observations.

        Args:
            scene_name (str): Scene name.
            start_cam (str, optional): Camera frame from which the rendering
                starts. Defaults to None, corresponding to the first frame.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        selected_scene = None
        start_idx = -1
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                selected_scene = scene
                if start_cam is not None:
                    start_idx = -1
                    for i, img in enumerate(scene['images']):
                        img_path = img['img_path']
                        if dataset == 'scannet':
                            cam_name = img_path.split('/')[-1][:-4]
                        elif dataset == '3rscan':
                            cam_name = img_path.split('/')[-1][:-10]
                        elif dataset == 'matterport3d':
                            cam_name = img_path.split(
                                '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                        else:
                            cam_name = img_path.split('/')[-1][:-4]
                        if cam_name == start_cam:
                            start_idx = i
                            break
                    if start_idx == -1:
                        print('No such camera')
                        return
                else:
                    start_idx = 0

        if selected_scene is None:
            print('No such scene')
            return

        drawer = ContinuousOccupancyDrawer(dataset, self.data_root[dataset],
                                           selected_scene, self.classes,
                                           self.id_to_index,
                                           self.color_selector, start_idx)
        drawer.begin()

    def render_occupancy(self, scene_name):
        """Render the occupancy annotation of a given scene.

        Args:
            scene_name (str): Scene name.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        if dataset == 'scannet':
            filepath = os.path.join(self.data_root['scannet'], 'scans', region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == '3rscan':
            filepath = os.path.join(self.data_root['3rscan'], region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.data_root['matterport3d'], building,
                                    'occupancy', f'occupancy_{region}.npy')
        else:
            raise NotImplementedError

        if self.verbose:
            print('Loading occupancy')
        gt_occ = np.load(filepath)
        if self.verbose:
            print('Loading complete')
        point_cloud_range = [-3.2, -3.2, -1.28 + 0.5, 3.2, 3.2, 1.28 + 0.5]
        # occ_size = [40, 40, 16]
        grid_size = [0.16, 0.16, 0.16]
        points = np.zeros((gt_occ.shape[0], 6), dtype=float)
        for i in range(gt_occ.shape[0]):
            x, y, z, label_id = gt_occ[i]
            label_id = int(label_id)
            label = 'object'
            if label_id == 0:
                label = 'object'
            else:
                label = self.classes[self.id_to_index[label_id]]
            color = self.color_selector.get_color(label)
            color = [x / 255.0 for x in color]
            points[i][:3] = [
                x * grid_size[0] + point_cloud_range[0] + grid_size[0] / 2,
                y * grid_size[1] + point_cloud_range[1] + grid_size[1] / 2,
                z * grid_size[2] + point_cloud_range[2] + grid_size[2] / 2
            ]
            points[i][3:] = color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=grid_size[0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([frame, voxel_grid])

    def show_image(self, dataset_name, scene_name, camera_name, render_box=True):
        """Render an ego-centric image view with annotations.

        Args:
            scene_name (str): Scene name.
            camera_name (str): The name of rendered camera frame.
            render_box (bool): Whether to render box annotations in the image.
                Defaults to False.
        """
        # dataset = scene_name.split('/')[0]
        select = None
        for scene in self.data[dataset_name]:
            if scene['sample_idx'] == scene_name:
                select = scene
        for camera in select['images']:
            img_path = camera['img_path']
            img_path = os.path.join(self.data_root[dataset_name],
                                    img_path[img_path.find('/') + 1:])
            if dataset_name == 'scannet':
                cam_name = img_path.split('/')[-1][:-4]
            elif dataset_name == '3rscan':
                cam_name = img_path.split('/')[-1][:-10]
            elif dataset_name == 'matterport3d':
                cam_name = img_path.split('/')[-1][:-8] + img_path.split(
                    '/')[-1][-7:-4]
            else:
                cam_name = img_path.split('/')[-1][:-4]

            if cam_name == camera_name:
                axis_align_matrix = select['axis_align_matrix']
                extrinsic = axis_align_matrix @ camera['cam2global']
                if 'cam2img' in camera:
                    intrinsic = camera['cam2img']
                else:
                    intrinsic = select['cam2img']
                img_drawer = ImageDrawer(img_path, verbose=self.verbose)
                if render_box:
                    if self.verbose:
                        print('Rendering box')
                    for i in camera['visible_instance_ids']:
                        instance = select['instances'][i]
                        box = _9dof_to_box(
                            instance['bbox_3d'], self.classes[self.id_to_index[
                                instance['bbox_label_3d']]],
                            self.color_selector)
                        label = self.classes[self.id_to_index[
                            instance['bbox_label_3d']]]
                        if label in ["object"]: #NOTE omit unlabeled bounding boxes
                            continue
                        # if label not in ["floor"]:
                        #     continue
                        color = self.color_selector.get_color(label)
                        img_drawer.draw_box3d(box,
                                              color,
                                              label,
                                              extrinsic=extrinsic,
                                              intrinsic=intrinsic)
                    if self.verbose:
                        print('Rendering complete')

                img_drawer.show()
                return

        print('No such camera')
        return


