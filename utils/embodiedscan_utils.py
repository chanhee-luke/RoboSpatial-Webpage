# Utils for EmbodiedScan datasets

from collections import defaultdict


def show_embodiedscan_image(loader, dataset_name, scene_name, image, annotations):
    # Get camera name
    img_path = image['img_path']
    cam_name = ""
    if dataset_name == 'scannet':
        cam_name = img_path.split('/')[-1][:-4]
    elif dataset_name == '3rscan':
        cam_name = img_path.split('/')[-1][:-10]
    elif dataset_name == 'matterport3d':
        cam_name = img_path.split(
            '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
    else:
        cam_name = img_path.split('/')[-1][:-4]

    vis_obj_names = []
    for i in image['visible_instance_ids']:
        instance = annotations[i]
        label = loader.classes[loader.id_to_index[instance['bbox_label_3d']]]
        vis_obj_names.append(label)

    print(f"Visible objects are {vis_obj_names}")
    loader.show_image(f"{dataset_name}", scene_name, cam_name)


## Remove duplicates, remove partially visible objects
def filter_embodiedscan_objs(loader, annotations, vis_objs, dataset_name):
    label_count = {}
    
    # First pass to count label occurrences
    for i in vis_objs:
        instance = annotations[i]
        label = loader.classes[loader.id_to_index[instance['bbox_label_3d']]]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    
    # Second pass to create new dictionary without duplicates
    filtered_vis_objs = {}
    env_objs = {}
    all_objs = {}
    names = []
    for i in vis_objs:
        instance = annotations[i]
        # if dataset_name in ["3rscan"]:
        #     R_90_clockwise = np.array([
        #         [0, 1, 0, 0],
        #         [-1, 0, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ])
        #     instance["axis_align_matrix"] = R_90_clockwise @ instance["axis_align_matrix"]

        label = loader.classes[loader.id_to_index[instance['bbox_label_3d']]]
        if label_count[label] == 1 and label not in ["wall", "ceiling", "floor", "object"]:
            filtered_vis_objs[i] = instance
            names.append(label)
        if label in ["floor", "wall"]:
            env_objs[label] = instance
        if label not in ["floor", "wall", "ceiling"]:
            all_objs[i] = instance

    
    return filtered_vis_objs, env_objs, all_objs, names


## Remove duplicates, remove partially visible objects
def filter_embodiedscan_objs_for_grounding(loader, annotations, vis_objs, dataset_name):
    label_count = {}
    
    # First pass to count label occurrences
    for i in vis_objs:
        instance = annotations[i]
        label = loader.classes[loader.id_to_index[instance['bbox_label_3d']]]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    
    # Second pass to create new dictionary without duplicates
    single_vis_objs = {}
    multi_vis_objs = defaultdict(list)
    env_objs = {}
    all_objs = {}
    names = set()
    for i in vis_objs:
        instance = annotations[i]
        # if dataset_name in ["3rscan"]:
        #     R_90_clockwise = np.array([
        #         [0, 1, 0, 0],
        #         [-1, 0, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ])
        #     instance["axis_align_matrix"] = R_90_clockwise @ instance["axis_align_matrix"]

        label = loader.classes[loader.id_to_index[instance['bbox_label_3d']]]
        if label_count[label] == 1 and label not in ["wall", "ceiling", "floor", "object"]:
            single_vis_objs[label] = instance
            names.add(label)
        if label_count[label] > 1 and label not in ["wall", "ceiling", "floor", "object"]:
            multi_vis_objs[label].append(instance)
            names.add(label)
        if label in ["floor", "wall"]:
            env_objs[label] = instance
        if label not in ["floor", "wall", "ceiling"]:
            all_objs[i] = instance
        


    
    return single_vis_objs, multi_vis_objs, env_objs, all_objs, names
