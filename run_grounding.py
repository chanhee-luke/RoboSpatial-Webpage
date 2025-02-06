# Main entry script to generate grounding data from indoor scan datasets
# Currently supported: Scannet, Matterport3d, 3rscan, Multiscan

import argparse
import yaml
import os
import json
from tqdm import tqdm
from collections import defaultdict

from grounding_generator import generate_and_save_grounding_ann
from embodiedscan_loader import EmbodiedScanLoader
from multiscan_loader import MultiScanLoader

# from multiscan import process_multiscan
# function_map = {
#     "multiscan": process_multiscan,
# }

def parse_args():
    parser = argparse.ArgumentParser(description="Parse configuration file for dataset processing.")
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file.')
    parser.add_argument('--range', type=int, nargs=2, help='Range specified as two integers: start and end (inclusive).')
    args = parser.parse_args()

    if args.range:
        start, end = args.range
        if start > end:
            parser.error("Start of range must not be greater than end.")
    
    return args

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run(config):

    # Load & read each image from dataset, parse into a unified format
    # Output needed for each dataset: image, depth, object names, 3D bounding boxes, 3D bounding box orientations (which direction its facing)
    dataset_list = config["datasets"].split(",")
    dataset_dict = {}
    for dataset_name in dataset_list:
        if dataset_name == "multiscan":
            dataset_dict[dataset_name] = {"dataset_path": config[dataset_name]["dataset_path"], "image_path": config[dataset_name]["image_path"]}
        else:
            dataset_dict[dataset_name] = config[dataset_name]["dataset_path"]
    
    splits = config["split"].split(",")
    embodiedscan_ann = list()
    for split in splits:
        embodiedscan_ann.append(config["embodiedscan_ann"][split])

    multiscan_loader = None
    embodiedscan_loader = None
    if "multiscan" in dataset_dict.keys():
        multiscan_loader = MultiScanLoader(dataset_dict.pop("multiscan"), verbose=config["verbose"])
    if any(s in dataset_dict.keys() for s in ["scannet", "matterport3d", "3rscan"]):
        embodiedscan_loader = EmbodiedScanLoader(data_root=dataset_dict, ann_file=embodiedscan_ann, verbose=config["verbose"])
    

    generated_something = False


    # Generate QA
    # For each image instance, generate QA
    ## For multiscan:
    ## image_ids contain only image names
    ## image_obj_ann contains camera poses for each image and OBB
    if multiscan_loader:
        for dataset_name, scene_id in multiscan_loader.list_scenes():
            # Retrieve annotation for the scene
            # if scene_id != "scene_00024_01":
            #     continue
            scene_obj_ann, image_ann_list = multiscan_loader.list_images_and_instances(scene_id)
            scene_stats = generate_and_save_ann(multiscan_loader, dataset_name, scene_id, image_ann_list, config, annotations=scene_obj_ann)


    ## For embodiedscan:
    ## image_ann_list contain image names and camera poses
    ## scene_obj_ann contains OBBs
    if embodiedscan_loader:
        if "multiscan" in dataset_list:
            dataset_list.remove("multiscan")
        # Iterate each scene
        num_images = 0
        completed_scenes = defaultdict(list)

        num_total_images = defaultdict(int)
        num_total_obj_bboxes = defaultdict(int)
        num_total_obj_cat_in_img = defaultdict(int)

        # # Check if the scene is already processed
        # for dataset, data in embodiedscan_loader.data.items():
        #     print(f"{dataset} {len(data)}")
        
        # exit()

        
        for dataset_name, idx, scene_name in tqdm(embodiedscan_loader.list_scenes(dataset_list), desc=f"Scene loop"):
            # Skip already processed scenes
            completed_scenes = {}
            file_path = os.path.join((config[dataset_name]["output_path"]), f"point_grounding_progress.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    completed_scenes[dataset_name] = json.load(file)


            if dataset_name in completed_scenes:
                if scene_name in completed_scenes[dataset_name]:
                    # print(f"{scene_name} already processed!")
                    continue

            # Only run selected scenes if requested
            if "range" in config:
                if idx not in config["range"]:
                    continue
            
            # if "scene0191_01" not in scene_name:
            #     continue

            print(f"Start processing {dataset_name} {scene_name}")

            # Main relationship generation block
            scene_obj_ann = embodiedscan_loader.list_instances(dataset_name, scene_name)
            image_ann_list = embodiedscan_loader.list_images(dataset_name, scene_name)
            scene_stats = generate_and_save_grounding_ann(embodiedscan_loader, dataset_name, scene_name, image_ann_list, config, annotations=scene_obj_ann)
            

            num_total_images[scene_stats["dataset_name"]] += scene_stats["num_total_images"]
            num_total_obj_bboxes[scene_stats["dataset_name"]] += scene_stats["num_total_obj_bboxes"]
            num_total_obj_cat_in_img[scene_stats["dataset_name"]] += scene_stats["num_total_obj_cat_in_img"]
            
            if dataset_name not in completed_scenes:
                completed_scenes[dataset_name] = []

            completed_scenes[dataset_name].append(scene_name)

            print(f"Done processing {dataset_name} {scene_name}")

            # Save stats for each scene
            # file_path = os.path.join((config[dataset_name]["output_path"]), f"grounding_ann_stats.jsonl")
            # with open(file_path, 'a') as file:
            #     json.dump(scene_stats, file)
            #     file.write('\n')
            
            # # Save progress stats
            # file_path = os.path.join((config[dataset_name]["output_path"]), f"grounding_progress.json")
            # with open(file_path, 'w') as file:
            #     json.dump(completed_scenes[dataset_name], file, indent=4)
            
            # print(f"Saved grounding stats for {dataset_name} {scene_name}")

            generated_something = True


        # if generated_something:
        #     print(f'num_total_obj_pairs[{scene_stats["dataset_name"]}] = {num_total_obj_pairs[scene_stats["dataset_name"]]}')
        #     print(f'num_total_images[{scene_stats["dataset_name"]}] = {num_total_images[scene_stats["dataset_name"]]}')
        #     print(f'num_relationships_pairs[{scene_stats["dataset_name"]}] = {num_relationships_pairs[scene_stats["dataset_name"]]}')
        #     print(f'num_compatibility_pairs[{scene_stats["dataset_name"]}] = {num_compatibility_pairs[scene_stats["dataset_name"]]}')

        # total_scene_ann_stats = {
        #     'dataset_name': dataset_name,
        #     'scene_name': scene_name,
        #     'num_total_obj_pairs': num_total_obj_pairs,
        #     'num_total_images': num_images,
        #     'num_relationships_pairs': num_total_obj_rel_pairs_in_img,
        #     'num_compatibility_pairs': num_total_obj_comp_pairs_in_img
        # }


    # Save stats
    # Save stats for each image in a json file
    return




if __name__ == "__main__":

    args = parse_args()
    config = load_config(args.config)
    if args.range:
        config["range"] = range(args.range[0], args.range[1]+1)
    
    run(config)
