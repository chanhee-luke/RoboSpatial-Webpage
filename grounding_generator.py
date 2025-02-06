# Script containing functions to generate QA from OBB annotations

import itertools
import os
import json
from collections import Counter
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm

from visualization.utils import _9dof_to_box
from utils.spatial_relationships import get_boxes_relationship, get_empty_space, can_fit_on_top, can_fit_object_a_in_relation_to_b, get_point_in_space_relative_to_object
from utils.obj_properties import items_with_face, flat_surface_items, movable_and_placeable_items
from utils.embodiedscan_utils import show_embodiedscan_image, filter_embodiedscan_objs, filter_embodiedscan_objs_for_grounding
from utils.multiscan_utils import create_oriented_bounding_box, filter_multiscan_objs, get_visible_boxes, show_multiscan_image
from utils.spatial_relationship_utils import get_box2d_coordinates_and_depth, project_points_to_image, generate_points_in_bounding_box, order_bbox_coords, clip_bounding_box, convert_to_square_bounding_box, get_box3d_coordinates, is_bounding_box_on_floor
from utils.vis_utils import show_image_with_points_and_bbox, show_image_with_bbox, show_image_with_points, show_image_with_bbox_points, show_image_with_square_bbox_points
from visualization.img_drawer import ImageDrawer

DEBUG=False



def generate_and_save_grounding_ann(loader, dataset_name, scene_name, image_list, config, annotations):
    # Generate QA for a scene, each scene has multiple images

    is_multiscan = False
    if dataset_name == "multiscan":
        is_multiscan = True

    coordinate_systems = config["qa_gen_options"]["coordinate_systems"]
    # Get camera intrinsic matrix if not already in annotation (e.g. scannet)
    # intrinsic_matrix = []
    # if dataset_name == "scannet":
    #     scene_path = scene_name.replace("scannet/", "posed_images/")
    #     intrinsic_path = os.path.join(config["scannet"]["dataset_path"], scene_path, "intrinsic.txt")
    #     intrinsic_matrix = np.loadtxt(intrinsic_path)
    
    # if dataset_name == "3rscan":
    #     scene_path = scene_name.replace("3rscan/", "")
    #     intrinsic_path = os.path.join(config["3rscan"]["dataset_path"], scene_path, "sequence", "_info.txt")
    #     def extract_calibration_color_intrinsic(file_path):
    #         with open(file_path, 'r') as file:
    #             lines = file.readlines()
            
    #         for line in lines:
    #             if "m_calibrationColorIntrinsic" in line:
    #                 intrinsic_values = line.split('=')[1].strip().split()
    #                 intrinsic_array = np.array(intrinsic_values, dtype=float).reshape(4, 4)
    #                 return intrinsic_array

    #     intrinsic_matrix = extract_calibration_color_intrinsic(intrinsic_path)

    # For stats
    num_images = 0

    num_total_obj_bboxes = 0
    num_total_obj_cat_in_img = 0

    # for image in tqdm(image_list, desc="Inner image loop", leave=False):
    for image in tqdm(image_list, leave=False):

        
        # if "340" not in image["img_path"]:
        #     continue
        
        # if num_images in range(0,8):
        #     num_images += 1
        #     continue
        # print(image)
        # print(annotations[0]["cam2img"])
        # exit()

        # Get image size
        if dataset_name == "3rscan":
            image_name = image["img_path"].replace("3rscan/", "")
            image_path = os.path.join(config["3rscan"]["dataset_path"], image_name)
        elif dataset_name == "matterport3d":
            image_name = image["img_path"].replace("matterport3d/", "")
            image_path = os.path.join(config["matterport3d"]["dataset_path"], image_name)
        elif dataset_name == "scannet":
            image_name = image["img_path"].replace("scannet/", "")
            image_path = os.path.join(config["scannet"]["dataset_path"], image_name)
        elif dataset_name == "multiscan":
            image_name = f"sampled_frame_at_{image}.png"
            image_path = os.path.join(config["multiscan"]["image_path"], scene_name, image_name)
        # Open the image using Pillow
        image_file = Image.open(image_path)
        # Get the dimensions
        width, height = image_file.size
        image_size = (width, height)





        # Get visible objects
        vis_objs = []
        if is_multiscan:
            vis_objs = get_visible_boxes(image_list[image], annotations, image_size)
        else:
            vis_objs = image['visible_instance_ids']
        
        
        
        if DEBUG:
            if is_multiscan:
                #TODO show image for multiscan
                a = 1
            else:
                # if image["img_path"] != 'scannet/posed_images/scene0119_00/00110.jpg':
                #     continue
                if "fcf66d9e-622d-291c-84c2-bb23dfe31327" in image["img_path"]:
                    continue
                show_embodiedscan_image(loader, dataset_name, scene_name, image, annotations)
                exit()
        
        # Remove duplicates and unnecessary object annotations (e.g. wall, ceiling, floor, object)
        filtered_vis_objs = {}
        if is_multiscan:
            filtered_vis_objs, env_objs, all_objs, names = filter_multiscan_objs(vis_objs, image_list[image], annotations, image_size)
            # print("filtered_vis_objs:", filtered_vis_objs)
            # print("env_objs:", env_objs)
            # print("all_objs:", all_objs)
            # print("names:", names)
            # exit()
        else:
            ## Remove cle{wall, ceiling, floor, object} for EmbodiedScan
            single_vis_objs, multi_vis_objs, env_objs, all_objs, names = filter_embodiedscan_objs_for_grounding(loader, annotations, vis_objs, dataset_name)
        

        if DEBUG:
            print(names)
            print(filtered_vis_objs.keys())

        # print("single vis", single_vis_objs)
        # print("multi vis", multi_vis_objs)
        # print(vis_objs)
        # print(names)
        # exit()
        

        #NOTE skip images that only look at the ceiling
        if "floor" not in env_objs:
            continue

        #NOTE Begin processing
        num_images += 1
    

        # wall_box = _9dof_to_box(env_objs["wall"]["bbox_3d"])
        # Create floor box if there is none (e.g. scannet)
        boxes = []
        names = []
        for i, obj in all_objs.items():
            boxes.append(_9dof_to_box(obj["bbox_3d"]))
            names.append(obj["name"])

        # test_boxes = []
        # for i, obj in all_objs.items():
        #     if obj["name"] in ["towel"]:
        #         test_boxes.append(_9dof_to_box(obj["bbox_3d"]))
        
        floor_box = _9dof_to_box(env_objs["floor"]["bbox_3d"])
        if dataset_name == "scannet":
            min_bound = np.min([box.get_min_bound() for box in boxes + [floor_box]], axis=0)
            max_bound = np.max([box.get_max_bound() for box in boxes + [floor_box]], axis=0)
            floor_box = [min_bound, max_bound]



        # print(get_empty_space(floor_box, boxes))
        # exit()
        obj_results = []
        num_obj_cat_in_img = 0
        num_obj_bbox_in_img = 0


        # Iterate visible objects and get relationship
        # for obj in tqdm(list(filtered_vis_objs.keys()), desc="Inner object loop", leave=False):


        

        for obj_name in tqdm(names, leave=False):

            obj_result = {
                f"{obj_name}": {}
            }

            # if obj_name not in flat_surface_items:
            #     continue

            # if "fcf66d9e-622d-291c-84c2-bb23dfe31327" in image["img_path"]:
            #     continue

            # if not "towel" == filtered_vis_objs[objs[0]]["name"] or not "sink" ==  filtered_vis_objs[objs[1]]["name"]:
            #     continue
            # if not "cabinet" ==  filtered_vis_objs[objs[1]]["name"]:
            #     continue

            is_single = True # Flag for generating point grounding
            obj = None
            if obj_name in single_vis_objs:
                obj = single_vis_objs[obj_name]
            if obj_name in multi_vis_objs:
                obj = multi_vis_objs[obj_name]
                is_single = False
            if not obj:
                continue
                # raise ValueError(f"{obj_name} does not exist! There must be something wrong....")
            
            if type(obj) != list:
                obj = [obj]
            
            # print(image['img_path'])

            # square_obj_bboxes, obj_bboxes = get_object_bbox_grounding(obj, image, coordinate_systems, image_size)

            if is_single:
                # obj = obj[0]
                # obj["box"] = _9dof_to_box(obj["bbox_3d"])
                # print(obj["name"], np.asarray(obj["box"].get_box_points()))
                # if is_bounding_box_on_floor(obj["box"]):
                #     print(f"{obj['name']} on floor")
                # else:
                #     print(f"{obj['name']} NOT on floor")

                obj_points = get_object_point_space_grounding(obj, image, coordinate_systems, floor_box, boxes, image_size)

            # For stats
            # num_obj_bbox_in_img += len(obj_bboxes)
            # num_obj_cat_in_img += 1

            # obj_result[f"{obj_name}"]["square_bbox"] = square_obj_bboxes
            # obj_result[f"{obj_name}"]["bbox"] = obj_bboxes
            # obj_result[f"{obj_name}"]["depth_bbox"] = obj_depths

            obj_results.append(obj_result)

            # import json
            # # Convert to JSON string with pretty printing
            # pretty_json = json.dumps(obj_results, indent=4)

            # # Print the pretty JSON
            # print(image["img_path"])
            # print(pretty_json)
            
        if not is_multiscan:
            if "cam2img" not in image:
                image["cam2img"] = annotations[0]["cam2img"]
            if "depth_cam2img" not in image:
                image["depth_cam2img"] = annotations[0]["depth_cam2img"]

            image["cam2global"] = image["cam2global"].tolist()
            image["cam2img"] = image["cam2img"].tolist()
            if type(image["depth_cam2img"]) != list:
                image["depth_cam2img"] = image["depth_cam2img"].tolist()

            image_results = {
                "dataset": dataset_name,
                "scene_name": scene_name,
                "image_path": image["img_path"],
                "depth_path": image["depth_path"],
                "camera_annotations": {
                    "cam2global": image["cam2global"],
                    "cam2img": image["cam2img"],
                    "depth_cam2img": image["depth_cam2img"]
                },
                "visible_instance_ids": image['visible_instance_ids'],
            }
        
        image_results["object_bboxes"] = obj_results

        img_path = os.path.join("/mnt/storage/datasets/scan_datasets/EmbodiedScan/data", image["img_path"])
        idwr = ImageDrawer(img_path)
        idwr.show()

        # Save annotations
        # scene_name = scene_name.replace(dataset_name, "")
        # scene_name = scene_name.strip("/")
        # folder_path = os.path.join((config[dataset_name]["output_path"]), scene_name)
        # image_name = os.path.basename(image["img_path"])
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # file_name = f"{image_name}.grounding.ann.json"
        # file_path = os.path.join(folder_path, file_name)
        # with open(file_path, 'w') as json_file:
        #     json.dump(image_results, json_file, indent=4)
        
        num_total_obj_bboxes += num_obj_bbox_in_img
        num_total_obj_cat_in_img += num_obj_cat_in_img



        # print(image_results)

    
    
    
    scene_ann_stats = {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'num_total_obj_bboxes': num_total_obj_bboxes,
        'num_total_images': num_images,
        'num_total_obj_cat_in_img': num_total_obj_cat_in_img
    }

    return scene_ann_stats


def get_object_bbox_grounding(objs, image_ann, coordinate_systems, image_size):

    square_obj_bboxes = []
    obj_bboxes = []
    obj_depths = []

    for obj in objs:
        obj["box"] = _9dof_to_box(obj["bbox_3d"])

        axis_align_matrix = obj['axis_align_matrix']
        extrinsic = axis_align_matrix @ image_ann['cam2global']
        if 'cam2img' in image_ann:
            intrinsic = image_ann['cam2img']
        else:
            intrinsic = obj['cam2img']
        
        # box1_coords, box1_depth = get_box2d_coordinates_and_depth(obj["box"], extrinsic, intrinsic)
        box1_coords = get_box3d_coordinates(obj["box"], extrinsic, intrinsic, image_size)

        if box1_coords is None: #NOTE if bbox is wrong
            continue

        # print(obj["name"], obj["box"], box1_coords)

        clipped_box1_coords = clip_bounding_box(box1_coords, image_size)

        
        sqaure_clipped_box1_coords = convert_to_square_bounding_box(clipped_box1_coords)


        # print(box1_coords)
        # print(clipped_box1_coords)
        # print(sqaure_clipped_box1_coords)
        # print(image_size)

        # img_path = os.path.join("/mnt/storage/datasets/scan_datasets/EmbodiedScan/data", image_ann["img_path"])
        # show_image_with_bbox(img_path, obj["box"], extrinsic, intrinsic)
        # show_image_with_bbox_points(img_path, clipped_box1_coords)
        # print(obj["name"])
        # show_image_with_square_bbox_points(img_path, sqaure_clipped_box1_coords)


        square_obj_bboxes.append(sqaure_clipped_box1_coords.tolist())
        obj_bboxes.append(clipped_box1_coords.tolist())

        # obj_depths.append(box1_depth)

        # print(obj["name"])
        # img_path = os.path.join("/mnt/storage/datasets/scan_datasets/EmbodiedScan/data", image_ann["img_path"])
        # idwr = ImageDrawer(img_path)
        # idwr.draw_box3d(obj["box"], [255, 255, 0], obj["name"], extrinsic, intrinsic)
        # idwr.show()
        # exit()

    return square_obj_bboxes, obj_bboxes


def get_object_point_space_grounding(objs, image_ann, coordinate_systems, floor_box, boxes, image_size, threshold=0.0):

    obj = objs[0]
    obj["box"] = _9dof_to_box(obj["bbox_3d"])

    have_face = False
    if obj["name"] in items_with_face:
        have_face = True

    axis_align_matrix = obj['axis_align_matrix']
    extrinsic = axis_align_matrix @ image_ann['cam2global']
    if 'cam2img' in image_ann:
        intrinsic = image_ann['cam2img']
    else:
        intrinsic = obj['cam2img']

    points_on_top = []
    point_on_space = {}
    
    if obj["name"] in flat_surface_items:
        # Get the eight vertices of the bounding box
        corners = np.asarray(obj["box"].get_box_points())

        z_coordinates = corners[:, 2]
        top_face_indices = np.argsort(z_coordinates)[-4:]  # Indices of the top 4 points

        # Get the coordinates of the top face points
        top_face_points = corners[top_face_indices]
        

        projected_points, projected_depth = project_points_to_image(top_face_points, extrinsic, intrinsic)
        ordered_projected_points = order_bbox_coords(projected_points)
        points = generate_points_in_bounding_box(ordered_projected_points, 10)

        points_on_top = points

    
    print(obj["name"])
    print(image_ann["img_path"])

    if is_bounding_box_on_floor(obj["box"]):
        points_on_space, points_pixel = get_point_in_space_relative_to_object(floor_box, boxes, obj["box"], extrinsic, intrinsic, image_size, have_face, num_samples=10, threshold=0.1, grid_resolution=0.1)
    
        print(points_on_space)
        img_path = os.path.join("/mnt/storage/datasets/scan_datasets/EmbodiedScan/data", image_ann["img_path"])
        for frame in points_on_space:
            print("all points:")
            show_image_with_points_and_bbox(img_path, obj["box"], points_pixel, extrinsic, intrinsic)
            for direction in points_on_space[frame]:
                print(frame, direction)
                show_image_with_points_and_bbox(img_path, obj["box"], points_on_space[frame][direction], extrinsic, intrinsic)

    # show_image_with_bbox(img_path, projected_points)
        
    return points_on_top, points_on_space



def check_object_compatibility_relationships(obj1, obj2, image_ann, coordinate_systems, floor_box, boxes, threshold=0.0):
    camera_centric = []
    worldcentric = []
    objectcentric = []

    obj1["box"] = _9dof_to_box(obj1["bbox_3d"])
    obj2["box"] = _9dof_to_box(obj2["bbox_3d"])

    axis_align_matrix = obj1['axis_align_matrix']
    extrinsic = axis_align_matrix @ image_ann['cam2global']
    if 'cam2img' in image_ann:
        intrinsic = image_ann['cam2img']
    else:
        intrinsic = obj1['cam2img']

    results = can_fit_object_a_in_relation_to_b(floor_box, boxes, obj1["box"] , obj2["box"], extrinsic, intrinsic, grid_resolution=0.1)
    # print("object compatibility")
    # print(obj1["name"], obj2["name"])
    # print(results)
    # print()

    fits_on_top = False
    if obj1["name"] not in movable_and_placeable_items or obj2["name"] not in flat_surface_items:
        fits_on_top = False
    else:
        fits_on_top = True if can_fit_on_top(obj1["box"], obj2["box"]) else False

    results["worldcentric"] = {}
    results["worldcentric"]["on_top"] = fits_on_top

    return results