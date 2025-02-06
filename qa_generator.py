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
from utils.spatial_relationships import get_boxes_relationship, get_empty_space, can_fit_on_top, can_fit_object_a_in_relation_to_b
from utils.obj_properties import items_with_face, flat_surface_items, movable_and_placeable_items
from utils.embodiedscan_utils import show_embodiedscan_image, filter_embodiedscan_objs
from utils.multiscan_utils import create_oriented_bounding_box, filter_multiscan_objs, get_visible_boxes, show_multiscan_image

DEBUG=False

# class QA_GEN:
#     def __init__(self, dataset: 'EmbodiedScanLoader | MultiScanLoader',
#                  config,
#                  verbose: bool = False,):
#         if not isinstance(dataset, (EmbodiedScanLoader, MultiScanLoader)):
#             raise TypeError(f"Expected an instance of Dataset or AnotherDataset, got {type(dataset).__name__}")
#         self.dataset = dataset

def generate_and_save_ann(loader, dataset_name, scene_name, image_list, config, annotations):
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

    num_total_obj_pairs = 0
    num_total_obj_rel_pairs_in_img = 0
    num_total_obj_comp_pairs_in_img = 0

    for image in tqdm(image_list, desc="Inner image loop", leave=False):
        


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
            filtered_vis_objs, env_objs, all_objs, names = filter_embodiedscan_objs(loader, annotations, vis_objs, dataset_name)
        

        if DEBUG:
            print(names)
            print(filtered_vis_objs.keys())
        
        # show_multiscan_image(config, scene_name, image, image_list[image], annotations)
        # exit()

        # Skip images with less than 4 unique objects
        # if len(set(vis_objs)) < 4:
        #     continue

        # if len(filtered_vis_objs) < 4:
        #     continue
        
        # if num_images < 20:
        #     num_images += 1
        #     continue
        # print(scene_name, image)
        # print(names)

        # show_multiscan_image(config, scene_name, image, image_list[image], annotations)
        # exit()

        # Getting empty space
        # print()
        # print(scene_name)
        # print(env_objs.keys())
        # print()

        #NOTE skip images that only look at the ceiling
        if "floor" not in env_objs:
            continue

        #NOTE Begin processing
        num_images += 1

        #NOTE Add intrinsic matrix if scannet
        # if dataset_name in ["scannet"]:
        #     image["cam2img"] = intrinsic_matrix
        
        # if dataset_name == "3rscan":
        #     image["cam2img"] = annotations[0]["cam2img"]
        
            # print(image_size)

            # # Define the extrinsic matrix (example values, replace with your actual matrix)
            # E = np.array([
            #     [0.866, -0.5, 0, 0],
            #     [0.5, 0.866, 0, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])

            # # Define the intrinsic matrix (example values, replace with your actual matrix)
            # K = np.array([
            #     [1000, 0, 320, 0],
            #     [0, 1000, 240, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])

            # Rotation matrix for 90 degrees clockwise
            # R_90_clockwise = np.array([
            #     [0, 1, 0, 0],
            #     [-1, 0, 0, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])

            # Apply the rotation to the extrinsic matrix
            # image["cam2global"] = R_90_clockwise @ image["cam2global"]

            # Modify the intrinsic matrix
            # K = image["cam2img"]
            # image["cam2img"] = np.array([
            #     [K[1, 1], 0, width - K[1, 2], 0],
            #     [0, K[0, 0], K[0, 2], 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])

            # print(image)

    

        # wall_box = _9dof_to_box(env_objs["wall"]["bbox_3d"])
        # Create floor box if there is none (e.g. scannet)
        boxes = []
        for i, obj in all_objs.items():
            boxes.append(_9dof_to_box(obj["bbox_3d"]))

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
        num_obj_pairs_in_img = 0
        num_obj_rel_pairs_in_img = 0
        num_obj_comp_pairs_in_img = 0
        # Iterate visible objects and get relationship
        for objs in tqdm(list(itertools.permutations(filtered_vis_objs.keys(), 2)), desc="Inner object loop", leave=False):
            obj1_name = filtered_vis_objs[objs[0]]["name"]
            obj2_name = filtered_vis_objs[objs[1]]["name"]
            obj_result = {
                f"{obj1_name}, {obj2_name}": {}
            }

            # if "fcf66d9e-622d-291c-84c2-bb23dfe31327" in image["img_path"]:
            #     continue

            # if not "towel" == filtered_vis_objs[objs[0]]["name"] or not "sink" ==  filtered_vis_objs[objs[1]]["name"]:
            #     continue
            # if not "cabinet" ==  filtered_vis_objs[objs[1]]["name"]:
            #     continue

            obj_localiztion_relationships = check_object_localization_relationships(filtered_vis_objs[objs[0]], filtered_vis_objs[objs[1]], image, coordinate_systems, image_size)
            num_obj_rel_pairs_in_img += 1
            if filtered_vis_objs[objs[0]]["name"] in movable_and_placeable_items:
                obj_compatability_relationships = check_object_compatibility_relationships(filtered_vis_objs[objs[0]], filtered_vis_objs[objs[1]], image, coordinate_systems, floor_box, boxes)
                num_obj_comp_pairs_in_img += 1
            else:
                obj_compatability_relationships = None


            obj_result[f"{obj1_name}, {obj2_name}"]["localization_relationships"] = obj_localiztion_relationships
            obj_result[f"{obj1_name}, {obj2_name}"]["compatability_relationships"] = obj_compatability_relationships
            obj_results.append(obj_result)

            num_obj_pairs_in_img += 1

            # import json
            # Convert to JSON string with pretty printing
            # pretty_json = json.dumps(obj_results, indent=4)

            # Print the pretty JSON
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
        
        image_results["object_relationships"] = obj_results

        # Save annotations
        scene_name = scene_name.replace(dataset_name, "")
        scene_name = scene_name.strip("/")
        folder_path = os.path.join((config[dataset_name]["output_path"]), scene_name)
        image_name = os.path.basename(image["img_path"])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = f"{image_name}.ann.json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as json_file:
            json.dump(image_results, json_file, indent=4)
        
        num_total_obj_pairs += num_obj_pairs_in_img
        num_total_obj_rel_pairs_in_img += num_obj_rel_pairs_in_img
        num_total_obj_comp_pairs_in_img += num_obj_comp_pairs_in_img



        # print(image_results)

    
    
    
    scene_ann_stats = {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'num_total_obj_pairs': num_total_obj_pairs,
        'num_total_images': num_images,
        'num_relationships_pairs': num_total_obj_rel_pairs_in_img,
        'num_compatibility_pairs': num_total_obj_comp_pairs_in_img
    }

    return scene_ann_stats




def check_object_localization_relationships(obj1, obj2, image_ann, coordinate_systems, image_size, threshold=0.0):

    # Cameracentric, Worldcentric, objectcentric
    # left/right, infront/behind, above/below
    # Cameracentric and worldcentric has same left/right, infront/behind, different above/below
    # Worldcentric and objectcentric has same left/right, above/below, different infront/behind

    # print(obj1, obj2)
    # print(image_ann)

    camera_centric = []
    worldcentric = []
    objectcentric = []

    # print(image_ann)

    obj1["box"] = _9dof_to_box(obj1["bbox_3d"])
    obj2["box"] = _9dof_to_box(obj2["bbox_3d"])

    axis_align_matrix = obj1['axis_align_matrix']
    extrinsic = axis_align_matrix @ image_ann['cam2global']
    if 'cam2img' in image_ann:
        intrinsic = image_ann['cam2img']
    else:
        intrinsic = obj1['cam2img'] 

    obj_localization_relationships = get_boxes_relationship(obj1, obj2, extrinsic, intrinsic, image_size)
    # print(obj1["name"], obj2["name"])
    # print(obj_localization_relationships)


    # print(f"Spatial relationship: {obj1['name']} is {horizontal_relation} and {depth_relation} relative to {obj2['name']} in the image")
    # print(f"Spatial relationship: {obj1['name']} is {vertical_world_relation} relative to {obj2['name']} in the world coordinates")
    # if obj2['name'] in items_with_face:
    #     print(f"Object-centric relationship: {obj1['name']} is {obj_centric_horizontal} and {obj_centric_depth} relative to {obj2['name']}")       
    return obj_localization_relationships


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