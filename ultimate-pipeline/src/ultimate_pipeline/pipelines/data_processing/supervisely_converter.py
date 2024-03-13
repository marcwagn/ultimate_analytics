"""Conversion from Supervisely to YOLO annotation format."""

import json
import pandas as pd
import numpy as np
from typing import Callable, Union, TypeAlias
import os
import logging

JSONGenerator: TypeAlias = Callable[[], dict]
# Python 3.12: type JSONGenerator = Callable[[],dict]

logger = logging.getLogger(__name__)


def convert_video_annotations(source: Union[str, dict]) -> pd.DataFrame:
    """
    Convert Supervisely video annotations to a DataFrame containing normalized bounding boxes.

    Args:
        source (str|dict) - file path to the Supervisely video annotations file, or a JSON dictionary

    Returns: Pandas DataFrame with the following columns:
        cls (int) - class id
        x (float) - bounding box centre x
        y (float) - bounding box centre x
        w (float) - bounding box width
        h (float) - bounding box width
        frame (str) - frame number or name

    References:
    - Supervisely format: https://developer.supervisely.com/getting-started/supervisely-annotation-format
    - YOLO v5 format: https://docs.ultralytics.com/datasets/detect/
    """
    annotations = None
    if source is None:
        raise ValueError("source argument is mandatory")
    elif isinstance(source, str):
        with open(source, 'r') as f:
            annotations = json.load(f)
    elif isinstance(source, dict):
        annotations = source
    else:
        raise ValueError("Unsupported type of source argument")

    logger.info("Reading Supervisely video annotations file")

    objects_list = annotations["objects"]
    if len(objects_list) == 0:
        return pd.DataFrame(columns=["cls", "x", "y", "w", "h"])
  
    objects_class_id_key_name = None
    figures_class_id_key_name = None
    if "key" in objects_list[0]:
        objects_class_id_key_name = "key"
        figures_class_id_key_name = "objectKey"
    elif "id" in objects_list[0]:
        objects_class_id_key_name = "id"
        figures_class_id_key_name = "objectId"
    else:
        raise ValueError("The JSON annotations file is expected to have either objects[...].key identifier, or objects[...].id")

    class_to_idx_map = {m[objects_class_id_key_name]: i for i, m in enumerate(objects_list)}

    def resolve_class_idx(fig):
        return class_to_idx_map[fig[figures_class_id_key_name]]

    # Each element in datas will correspond to 1 bounding box
    datas = []
    (width, height) = annotations["size"]["width"], annotations["size"]["height"]

    idx = 0
    for frame in annotations["frames"]:
        frame_index = frame["index"]
        for fig in frame["figures"]:
            class_id = resolve_class_idx(fig)
        
            (x1, y1) = fig["geometry"]["points"]["exterior"][0]
            (x2, y2) = fig["geometry"]["points"]["exterior"][1]

            box_arr = np.array([x1, y1, x2, y2], dtype='float')
            box_scaled = _xyxy2xywhn(box_arr, w=width, h=height)

            data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index, object_key=fig.get("objectKey", ""), x1=x1, x2=x2, y1=y1, y2=y2, idx=idx)
            # data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
            datas.append(data)
            idx += 1

    return pd.DataFrame(datas)

def convert_images_annotations_folder(source: Union[str, dict[str, JSONGenerator]], meta_file: Union[str, dict]) -> pd.DataFrame:
    """
    Convert a Supervisely image annotations in a folder to normalized bounding boxes.

    Args:
        source (str|JSONGenerator): path to the root folder containing JSON annotation files, 
            or a dictionary (keyed by file name) containing Callables that generate JSON annotation content as dict
        meta_file (str|dict): path to the meta.json file, or the JSON content of the meta file as dict
    """
    # Each DataFrame in dfs will correspond to bounding boxes from 1 image
    dfs = []

    annotations_generators = _source_to_annotation_generators(source)
    
    if isinstance(meta_file, str):
        meta_file_content = _read_json_content(meta_file)
    elif isinstance(meta_file, dict):
        meta_file_content = meta_file
    else:
        raise ValueError("meta_file needs to be a str or a JSON dict")
            
    meta_map = {m["id"]: i for i, m in enumerate(meta_file_content["classes"])}

    for key, annotations_gen in annotations_generators.items():
        dfs.append(convert_single_image_annotation_file(annotations_gen(), key, meta_map))
    return pd.concat(dfs, axis=0)


def convert_single_image_annotation_file(annotations: dict, frame_key: str, meta_map: dict) -> pd.DataFrame:
    """
    Convert a Supervisely annotation for a single image to normalized bounding boxes.
    Args:
        annotations (dict): content of JSON annotation file
        frame_key (str): frame number or name
        meta_map (dict): mapping between detection class ids to indexes
    """
    # Each element in datas correspond to 1 bounding box
    datas = []
    (width, height) = annotations["size"]["width"], annotations["size"]["height"]

    def resolve_class_idx(fig):
        return meta_map[fig["classId"]]

    idx = 0
    for detected in annotations["objects"]:
        # Skip non-rectangular annotations - they don't map to bounding boxes
        if detected["geometryType"] != 'rectangle':
            continue

        class_id = resolve_class_idx(detected)
            
        (x1, y1) = detected["points"]["exterior"][0]
        (x2, y2) = detected["points"]["exterior"][1]

        box_arr = np.array([x1, y1, x2, y2], dtype='float')
        box_scaled = _xyxy2xywhn(box_arr, w=width, h=height)

        data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_key, object_key="", x1=x1, x2=x2, y1=y1, y2=y2, idx=idx)
        # data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
        datas.append(data)
        idx += 1
    return pd.DataFrame(datas)

def convert_images_annotations_folder_to_pose_estimation(source: Union[str, JSONGenerator], meta_file: Union[str, dict]) -> pd.DataFrame:
    """
    Convert a Supervisely image annotations in a folder to to pose estimation DataFrame.

    Args:
        source (str|JSONGenerator): path to the root folder containing JSON annotation files, 
            or a dictionary (keyed by file name) containing Callables that generate JSON annotation content as dict
        meta_file (str|dict): path to the meta.json file, or the JSON content of the meta file as dict
    """
    # Each DataFrame in dfs will correspond to bounding boxes from 1 image
    dfs = []

    annotations_generators = _source_to_annotation_generators(source)

    if isinstance(meta_file, str):
        meta_file_content = _read_json_content(meta_file)
    elif isinstance(meta_file, dict):
        meta_file_content = meta_file
    else:
        raise ValueError("meta_file needs to be a str or a JSON dict")
    
    for key, annotations_gen in annotations_generators.items():
        dfs.append(convert_single_image_annotation_file_to_pose_estimation(annotations_gen(), key, meta_map=meta_file_content))
    return pd.concat(dfs, axis=0)

def convert_single_image_annotation_file_to_pose_estimation(annotations: dict, frame_key: str, meta_map: dict) -> pd.DataFrame:
    """
    Convert a Supervisely annotation for a single image to pose estimation DataFrame.

    Args:
        annotations (dict): content of JSON annotation file
        frame_key (str): frame number or name
        meta_file (dict): content of meta.json
    References:
        https://docs.ultralytics.com/datasets/pose/
    Examples:
        <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
        <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <pn-visibility>
    """
    if annotations is None or not isinstance(annotations, dict):
        raise ValueError("")

    # Each element in datas correspond to 1 bounding box
    datas = []
    (width, height) = annotations["size"]["width"], annotations["size"]["height"]

    graph_classes = list(filter(lambda c: c["shape"] == "graph", meta_map["classes"]))
    graph_id_to_idx = {}
    graph_id_to_nodes = {}
    for idx, g in enumerate(graph_classes):
        graph_id_to_idx[g["id"]] = idx
        graph_id_to_nodes[g["id"]] = g["geometry_config"]["nodes"]

    idx = 0
    for detected in annotations["objects"]:
        # Skip non-rectangular annotations - they don't map to bounding boxes
        if detected["geometryType"] != 'graph':
            continue

        class_id = detected["classId"]
        class_idx = graph_id_to_idx[class_id]
        
        min_x, min_y, max_x, max_y = width, height, 0, 0
        detected_nodes = detected["nodes"]
        for key, node in detected_nodes.items():
            if "disabled" in node:
                continue
            min_x = min(min_x, node["loc"][0])
            max_x = max(max_x, node["loc"][0])
            min_y = min(min_y, node["loc"][1])
            max_y = max(max_y, node["loc"][1])

        def scale_point(x,y):
            return x/width, y/height

        num_nodes = len(graph_id_to_nodes[class_id])
        # A matrix (num_classes * 3) of key points
        # Each row represents a triplet: (x, y, visibility), where x and y are scaled
        output_coords_triplets = np.zeros(shape=(num_nodes, 3), dtype='float')
        for i, node_key in enumerate(graph_id_to_nodes[class_id]):
            if node_key in detected_nodes:
                # visibility: 0 if absent, 1 if disabled, 2 if visible
                visibility =  1 if "disabled" in detected_nodes[node_key] else 2
                scaled_x, scaled_y = scale_point(x=detected_nodes[node_key]["loc"][0], y=detected_nodes[node_key]["loc"][1])
                output_coords_triplets[i] = np.array([scaled_x, scaled_y, visibility])

        output_coords_flattened = np.reshape(output_coords_triplets, (num_nodes*3))

        box_scaled_arr = _xyxy2xywhn(np.array([min_x, min_y, max_x, max_y], dtype='float'), h=height, w=width)

        vector = np.concatenate((np.array([class_idx]), box_scaled_arr, output_coords_flattened), axis=0)
        matrix = np.reshape(vector, (1, len(vector)))

        def generate_column_names():
            yield "cls"
            yield "x"
            yield "y"
            yield "w"
            yield "h"
            for i in range(len(output_coords_triplets)):
                yield f"px{i}"
                yield f"py{i}"
                yield f"p{i}_vis"
            #yield "frame"

        df = pd.DataFrame(matrix, columns=list(generate_column_names()))
        df["frame"] = frame_key
        #data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_key, object_key="", x1=x1, x2=x2, y1=y1, y2=y2, idx=idx)
        # data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
        datas.append(df)
        idx += 1
    return pd.concat(datas)

def _xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.
    NB: A simplified copy of ultralytics.utils.ops.xyxy2xywhn

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640

    Returns:
    y (np.ndarray):  The bounding box coordinates in (x, y, width, height, normalized) format
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def _source_to_annotation_generators(source: Union[str, dict[str, JSONGenerator]]) -> dict[str, JSONGenerator]:
    """
    Convert source to a dictionary of JSON generators.

    Args:
        source (str|JSONGenerator):  path to the root folder containing JSON annotation files, 
            or a dictionary (keyed by file name) containing Callables that generate JSON annotation content as dict
    """
    def list_files_only(folder):
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                yield file

    if source is None:
        raise ValueError("source argument is mandatory")
    elif isinstance(source, str):
        if not os.path.isdir(source) or not os.path.exists(source):
            raise ValueError("If source is passed as string, it needs to represent an existing directory")
        annotations_generators = {file: (lambda: _read_json_content(file, root=source)) for file in list_files_only(source)}
    elif isinstance(source, dict):
        if not all([isinstance(o, Callable) for o in source.values()]):
            raise ValueError("If source is passed as a dict, it needs to be a dict (keyed by file name) of callables returning JSON dicts")
        annotations_generators = source
    else:
        raise ValueError("Unsupported type of source argument")
    return annotations_generators

def _read_json_content(file_path, root=None):
    """Read a JSON file"""

    filepath = file_path if root is None else os.path.join(root, file_path)
    with open(filepath, 'r') as f:
        return json.load(f)
        