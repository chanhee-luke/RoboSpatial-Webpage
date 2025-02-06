# Script to sample RGB-D images from MultiScan video

import argparse
import yaml
import re
import os
import logging
import cv2
import random
import pandas as pd
import json
from tqdm import tqdm

from multiscan_utils import io

random.seed(42)

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse configuration file for dataset processing.")
    parser.add_argument('config_file', type=str, help='Path to the configuration YAML file.')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def visualize_frames(frames_dict):
    plt.figure(figsize=(15, 5))
    for i, (frame_index, frame) in enumerate(frames_dict.items()):
        plt.subplot(1, len(frames_dict), i+1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_index}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def sample_random_frames(video_path, sampling_ratio):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate random frame indices
    sample_size = int(total_frames * sampling_ratio)
    random_frame_indices = random.sample(range(total_frames), sample_size)
    
    sampled_frames = {}
    
    for frame_index in sorted(random_frame_indices):
        # Set the video position to the random frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        success, frame = video_capture.read()
        
        if success:
            sampled_frames[frame_index] = frame
    
    # Release the video capture object
    video_capture.release()
    
    return sampled_frames

def process_scan(scan, config):

    # Load annotation files
    with open(os.path.join(scan.scanPath, f"{scan.scanId}.annotations.json"), "r") as file:
        obb_annotations = json.load(file)
    
    camera_intrinsics = {}
    idx = 0
    with open(os.path.join(scan.scanPath, f"{scan.scanId}.jsonl"), "r") as file:
        for line in file:
            camera_intrinsic = json.loads(line)
            camera_intrinsics[idx] = camera_intrinsic
            idx += 1


    # Read the video file for the scan, split video file into individual frames and sample based on the sampling ratio and # of objects
    sampled_frames = sample_random_frames(os.path.join(scan.scanPath, f"{scan.scanId}.mp4"), sampling_ratio=config["multiscan"]["dataset_properties"]["sampling_ratio"])

    # Process sampled frames and save as images and print frame indices
    for frame_index, frame in sampled_frames.items():
        #TODO maybe For each image, identify the 3D bounding boxes and direction it faces

        # Save frames
        save_path = f'{config["multiscan"]["image_path"]}/posed_images/{scan.scanId}/sampled_frame_at_{frame_index}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

    # return everything
    return scan.scanId, sampled_frames.keys()

def process_multiscan(config):
    scan_paths = io.get_folder_list(config["multiscan"]["dataset_path"], join_path=True)

    scans_df = []
    for scan_path in scan_paths:
        try:
            scan_id = re.findall(r"scene\_[0-9]{5}\_[0-9]{2}", scan_path)[0]
        except:
            continue
        scene_id = '_'.join(scan_id.split('_')[:-1])
        row = pd.DataFrame([[scene_id, scan_id, scan_path]], columns=['sceneId', 'scanId', 'scanPath'])
        scans_df.append(row)
    scans_df = pd.concat(scans_df, ignore_index=True)

    scene_ids = scans_df['sceneId'].unique()

    seen_scans = set()
    with open(os.path.join(config["multiscan"]["image_path"], "posed_images", "stats.jsonl"), 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            seen_scans.update(json_obj.keys())
                    
    
    for scene_id in tqdm(scene_ids):
        scans = scans_df[scans_df['sceneId'] == scene_id]
        for i, scan in scans.iterrows():
            if scan.scanId in seen_scans:
                print(f"Skipping {scan.scanId}")
                continue
            log.info(f'Processing scan {scan.scanId}')
            _, sampled_frames_list = process_scan(scan, config)
            # Continously save stats
            with open(os.path.join(config["multiscan"]["image_path"], "posed_images", "stats.jsonl"), 'a') as file:
                json_object = json.dumps({scan.scanId: list(sampled_frames_list)})
                file.write(json_object + '\n')

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_file)
    process_multiscan(config)