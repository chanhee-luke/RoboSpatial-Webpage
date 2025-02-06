import datetime
import json
import hashlib
import os
import re
import shutil
import subprocess as subp
import sys
import traceback
import psutil
import resource
import multiprocessing
import multiprocessing.pool
import zlib
import tempfile
import cv2

from timeit import default_timer as timer
from enum import Enum

from utils.spatial_relationship_utils import get_box2d_coordinates_and_depth
from visualization.img_drawer import ImageDrawer

def file_exist(file_path, ext=''):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    elif ext in os.path.splitext(file_path)[1] or not ext:
        return True
    return False


def is_non_zero_file(file_path):
    return True if os.path.isfile(file_path) and os.path.getsize(file_path) > 0 else False


def file_extension(file_path):
    return os.path.splitext(file_path)[1]


def folder_exist(folder_path):
    if not os.path.exists(folder_path) or os.path.isfile(folder_path):
        return False
    else:
        return True


def ensure_dir_exists(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError:
        raise


def make_clean_folder(path_folder):
    try:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        else:
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
    except OSError:
        if not os.path.isdir(path_folder):
            raise


def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order

    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list, [0]

    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]

    indices = [i[0]
               for i in sorted(enumerate(file_list), key=lambda x: alphanum_key(x[1]))]
    return sorted(file_list, key=alphanum_key), indices

def get_file_list(path, ext='', join_path=True):
    file_list = []
    if not os.path.exists(path):
        return file_list

    for filename in os.listdir(path):
        file_ext = file_extension(filename)
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            if join_path:
                file_list.append(os.path.join(path, filename))
            else:
                file_list.append(filename)
    file_list, _ = sorted_alphanum(file_list)
    return file_list

def get_folder_list(path, join_path=True):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    folder_list = []
    for foldername in os.listdir(path):
        if not os.path.isdir(os.path.join(path, foldername)):
            continue
        if join_path:
            folder_list.append(os.path.join(path, foldername))
        else:
            folder_list.append(foldername)
    folder_list, _ = sorted_alphanum(folder_list)
    return folder_list


def filesize(file_path):
    if os.path.isfile(file_path):
        return os.path.getsize(file_path)
    else:
        return 0


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def write_json(data, filename, indent=2):
    if folder_exist(os.path.dirname(filename)):
        with open(filename, "w+") as fp:
            json.dump(data, fp, indent=indent)
    if not file_exist(filename):
        raise OSError('Cannot create file {}!'.format(filename))


def read_json(filename):
    if file_exist(filename):
        with open(filename, "r") as fp:
            data = json.load(fp)
        return data

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # print(f'memory limit {soft} TO {hard}')
    # print(f'memory limit set to {maxsize} GB')
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

# TODO: make memory limitation configurable
def call(cmd, log, rundir='', env=None, desc=None, cpu_num=0, mem=48, print_at_run=True, test_mode=False):
    if not cmd:
        log.warning('No command given')
        return 0
    if test_mode:
        log.info('Running ' + str(cmd))
        return -1
    cwd = os.getcwd()
    res = -1
    prog = None

    # constraint cpu usage with taskset
    if cpu_num > 0:
        all_cpus = list(range( min(psutil.cpu_count(), cpu_num)))
        sub_cpus = all_cpus[:cpu_num]
        str_cpus = ','.join(str(e) for e in sub_cpus)
        taskset_cmd = ['taskset', '-c', str_cpus]
        cmd = taskset_cmd + cmd
    
    try:
        start_time = timer()
        if rundir:
            os.chdir(rundir)
            log.info('Currently in ' + os.getcwd())
        log.info('Running ' + str(cmd))
        log.info(f'memory limit set to {mem} GB')
        setlimits = lambda: limit_memory(mem*1000*1000*1000) # in GB
        prog = subp.Popen(cmd, stdout=subp.PIPE, stderr=subp.STDOUT, env=env, preexec_fn=setlimits)
        # print output during the running
        if print_at_run:
            while True:
                nextline = prog.stdout.readline()
                if nextline == b'' and prog.poll() is not None:
                    break
                sys.stdout.write(nextline.decode("utf-8"))
                sys.stdout.flush()
        
        out, err = prog.communicate()
        if out:
            log.info(out.decode("utf-8"))
        if err:
            log.error('Errors reported running ' + str(cmd))
            log.error(err.decode("utf-8"))
        end_time = timer()
        delta_time = end_time - start_time
        desc_str = desc + ', ' if desc else ''
        desc_str = desc_str + 'cmd="' + str(cmd) + '"'
        log.info('Time=' + str(datetime.timedelta(seconds=delta_time)) + ' for ' + desc_str)
        res = prog.returncode
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt")
    except Exception as e:
        if prog is not None:
            prog.kill()
            out, err = prog.communicate()
        log.error(traceback.format_exc())
    os.chdir(cwd)
    return res

# https://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
def md5(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(blocksize), b''):
            hash.update(chunk)
    return hash.hexdigest()

# http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def natural_size(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class TIMEOUT(Enum):
    SECOND = 1
    MINUTE = 60
    HOUR = 3600

# reference https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    _wrap_exception = True

    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
        

# set mem limit for program
# reference https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
def set_memory_limit(percentage: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            set_memory_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                sys.stderr.write('\n\nERROR: Memory Exception, remaining memory  %.2f GB\n' % mem)
                sys.exit(1)
        return wrapper
    return decorator

@staticmethod
def decompress(input, output):
    d = zlib.decompressobj(-zlib.MAX_WBITS)
    chuck_size = 4096 * 4
    tmp = tempfile.NamedTemporaryFile(
        prefix=os.path.basename(input), dir=output, delete=False)
    logging.debug('temp file name: ' + tmp.name)

    with open(input, 'rb') as f:
        buffer = f.read(chuck_size)
        while buffer:
            outstr = d.decompress(buffer)
            tmp.write(outstr)
            buffer = f.read(chuck_size)

        tmp.write(d.flush())
        tmp.close()

    return tmp

import open3d as o3d
import numpy as np


# Turn multiscan obj annotation to open3d.geometry.OrientedBoundingBox
def create_oriented_bounding_box(obj):
    """
    Create an open3d.geometry.OrientedBoundingBox from the given annotation.

    Parameters:
        annotation (dict): Dictionary containing the 3D bounding box annotation.

    Returns:
        o3d.geometry.OrientedBoundingBox: The oriented bounding box.
    """
    # print(annotation)
    # Extracting data from the annotation
    centroid = obj['obb']['centroid']
    axes_lengths = obj['obb']['axesLengths']
    normalized_axes = obj['obb']['normalizedAxes']

    # Reshaping normalized axes into a 3x3 rotation matrix
    rotation_matrix = np.array(normalized_axes).reshape(3, 3)

    # Creating the OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center=centroid, R=rotation_matrix, extent=axes_lengths)
    
    return obb

def is_box_visible(box, intrinsic_matrix, extrinsic_matrix, image_width, image_height):
    # Transform the bounding box to camera coordinates
    box_in_camera_coords = box.translate(-box.center, relative=True)
    box_in_camera_coords.rotate(extrinsic_matrix[:3, :3], center=(0, 0, 0))
    box_in_camera_coords.translate(extrinsic_matrix[:3, 3], relative=True)

    # Get the corners of the bounding box
    corners = np.asarray(box_in_camera_coords.get_box_points())

    # Project corners to 2D image coordinates
    projected_corners = []
    for corner in corners:
        homogeneous_corner = np.append(corner, 1)
        corner_camera_coords = np.dot(extrinsic_matrix, homogeneous_corner)
        if corner_camera_coords[2] <= 0:  # Ignore points behind the camera
            continue
        corner_image_coords = np.dot(intrinsic_matrix, corner_camera_coords[:3])
        corner_image_coords /= corner_image_coords[2]  # Normalize by the depth
        projected_corners.append(corner_image_coords[:2])

    if not projected_corners:
        return False
    # Check if any projected corner is within the image boundaries
    for x, y in projected_corners:
        if 0 <= x < image_width and 0 <= y < image_height:
            return True

    return False

def check_visible_bounding_boxes(obbs, intrinsic_matrix, extrinsic_matrix, image_size):
    visible_boxes = []
    image_width, image_height = image_size

    for obj in obbs:
        try:
            box = create_oriented_bounding_box(obj)
        except:
            # print(f"{obj['label']} does not have a box!")
            continue
        
        corner_pixel, corner_depth = get_box2d_coordinates_and_depth(box, extrinsic_matrix, intrinsic_matrix)

        # print(obj['label'], corner_pixel)
        if is_object_in_image(corner_pixel, image_size[0], image_size[1]):
            visible_boxes.append(obj)

        # if is_box_visible(box, intrinsic_matrix, extrinsic_matrix, image_width, image_height):
        #     visible_boxes.append(obj["label"])

    return visible_boxes

def is_object_in_image(projected_corners, image_width, image_height):
    for coord in projected_corners:
        x, y = coord
        if 0 <= x < image_width and 0 <= y < image_height:
            return True
    return False

def get_visible_boxes(image, obbs, image_size):
    # First get visible objects
    extrinsic_matrix = image["cam2global"]
    intrinsic_matrix = image["cam2img"]

    visible_boxes = check_visible_bounding_boxes(obbs, intrinsic_matrix, extrinsic_matrix, image_size)

    return visible_boxes

## Remove duplicates, remove partially visible objects
def filter_multiscan_objs(vis_objs, image, obbs, image_size):

    label_count = {}
    
    # First pass to count label occurrences
    for obj in vis_objs:
        label = obj["label"]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    
    # Second pass to create new dictionary without duplicates
    filtered_vis_objs = {}
    env_objs = {}
    all_objs = {}
    names = []
    all_env_obj_pattern = re.compile(r'\b(wall|ceiling|floor|object|remove)\b')
    env_obj_pattern = re.compile(r'\b(wall|floor)\b')
    all_obj_pattern = re.compile(r'\b(wall|ceiling|floor)\b')
    for obj in vis_objs:
        label = obj["label"]
        if label_count[label] == 1 and not all_env_obj_pattern.search(label):
            filtered_vis_objs[label] = obj
            names.append(label)
        if all_obj_pattern.search(label):
            env_objs[label] = obj
        if not all_obj_pattern.search(label):
            all_objs[label] = obj

    
    return filtered_vis_objs, env_objs, all_objs, names


def show_multiscan_image(config, scene_name, image_id, image, annotations, render_box=True):
    """Render an ego-centric image view with annotations.

    Args:
        scene_name (str): Scene name.
        camera_name (str): The name of rendered camera frame.
        render_box (bool): Whether to render box annotations in the image.
            Defaults to False.
    """

    image_name = f"sampled_frame_at_{image_id}.png"
    image_path = os.path.join(config["multiscan"]["image_path"], scene_name, image_name)

    print(scene_name)
    print(image)

    # dataset = scene_name.split('/')[0]
    # axis_align_matrix = select['axis_align_matrix']
    # extrinsic = axis_align_matrix @ camera['cam2global']
    extrinsic = image['cam2global']
    intrinsic = image['cam2img']


    img_drawer = ImageDrawer(image_path, verbose=True)
    for obj in annotations:
        try:
            box = create_oriented_bounding_box(obj)
        except:
            # print(f"{obj['label']} does not have a box!")
            continue
        label = obj["label"]
        import random
        img_drawer.draw_box3d(box,
                            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                            label,
                            extrinsic=extrinsic,
                            intrinsic=intrinsic
                            )


    img_drawer.show()

    # if 'cam2img' in camera:
    #     intrinsic = camera['cam2img']
    # else:
    #     intrinsic = select['cam2img']
    # img_drawer = ImageDrawer(img_path, verbose=self.verbose)
    # if render_box:
    #     for i in camera['visible_instance_ids']:
    #         instance = select['instances'][i]
    #         box = _9dof_to_box(
    #             instance['bbox_3d'], self.classes[self.id_to_index[
    #                 instance['bbox_label_3d']]],
    #             self.color_selector)
    #         label = self.classes[self.id_to_index[
    #             instance['bbox_label_3d']]]
    #         if label in ["object"]: #NOTE omit unlabeled bounding boxes
    #             continue
    #         # if label not in ["floor"]:
    #         #     continue
    #         color = self.color_selector.get_color(label)
    #         img_drawer.draw_box3d(box,
    #                                 color,
    #                                 label,
    #                                 extrinsic=extrinsic,
    #                                 intrinsic=intrinsic)
    #     if self.verbose:
    #         print('Rendering complete')

    # img_drawer.show()
    # return

    print('No such camera')
    return

def project_bounding_boxes_to_image(obbs, image, intrinsic, extrinsic):
    """
    Projects a list of oriented bounding boxes (OBBs) onto an image.

    Parameters:
    - obbs: List of open3d.geometry.OrientedBoundingBox objects
    - image: The image onto which to project the bounding boxes (numpy array)
    - intrinsic: The camera intrinsic matrix (4x4 numpy array)
    - extrinsic: The camera extrinsic matrix (4x4 numpy array)

    Returns:
    - image_with_boxes: The image with projected bounding boxes drawn on it
    """

    def project_point(point, intrinsic, extrinsic):
        """ Projects a 3D point to 2D using intrinsic and extrinsic matrices """
        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(np.asarray(point), 1.0)
        # Transform point to camera coordinate system
        point_camera = np.dot(extrinsic, point_homogeneous)
        # Project point onto 2D image plane
        point_image = np.dot(intrinsic, point_camera[:3])
        point_image /= point_image[2]
        return int(point_image[0]), int(point_image[1])

    # Make a copy of the image to draw on
    image_with_boxes = image.copy()

    for obb in obbs:
        # Get the 8 corners of the OBB
        try:
            box = create_oriented_bounding_box(obj)
        except:
            # print(f"{obj['label']} does not have a box!")
            continue

        corners = np.asarray(box.get_box_points())

        # Project each corner to 2D
        corners_2d = [project_point(corner, intrinsic, extrinsic) for corner in corners]

        # Draw the OBB edges on the image
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for start, end in edges:
            cv2.line(image_with_boxes, corners_2d[start], corners_2d[end], (0, 255, 0), 2)

    return image_with_boxes

# Example usage
def show_multiscan_image_new(config, scene_name, image_id, image, obbs):

    image_name = f"sampled_frame_at_{image_id}.png"
    image_path = os.path.join(config["multiscan"]["image_path"], scene_name, image_name)

    print(scene_name)
    print(image)

    # dataset = scene_name.split('/')[0]
    # axis_align_matrix = select['axis_align_matrix']
    # extrinsic = axis_align_matrix @ camera['cam2global']
    extrinsic = image['cam2global']
    extrinsic = np.linalg.inv(extrinsic)
    intrinsic = image['cam2img']

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Project the bounding boxes onto the image
    image_with_boxes = project_bounding_boxes_to_image(obbs, image, intrinsic, extrinsic)

    # Display the image with bounding boxes
    cv2.imshow('Image with OBBs', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()