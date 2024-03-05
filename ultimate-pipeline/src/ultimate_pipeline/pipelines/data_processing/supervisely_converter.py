import json
import pandas as pd
import numpy as np
from typing import Union
import os
import logging

logger = logging.getLogger(__name__)

def convert_video_annotations(source: Union[str|dict]) -> pd.DataFrame:
    """
    Convert Supervisely video annotations to a DataFrame containing normalized bounding boxes.

    Args:
        source (str|DataFrame) - file path to the Supervisely video annotations file, or a JSON dictionary

    Returns: Pandas DataFrame with the following columns:
        cls (int) - class id
        x (float) - bounding box centre x
        y (float) - bounding box centre x
        w (float) - bounding box width
        h (float) - bounding box width
        frame (str) - frame number or name

    References:
    - Supervisely format: https://developer.supervisely.com/getting-started/supervisely-annotation-format
    - YOLO v5 format: https://docs.ultralytics.com/datasets/detect/p
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

    objects_map = annotations["objects"]
    if len(objects_map) == 0:
        return pd.DataFrame(columns=["cls", "x", "y", "w", "h"])
    
    # def make_map_to_index(json_objs:list, key="id"):
    #     mmap = { m[key]: i for i, m in enumerate(json_objs)}
    #     mmap = {}
    #     for i, m in enumerate(json_objs):
    #         mmap[m[key]] = i

    class_to_idx_map = {}
    resolve_class_idx = None
    if "key" in objects_map[0]:
        for i, m in enumerate(objects_map):
            class_to_idx_map[m["key"]] = i
        resolve_class_idx = lambda fig: class_to_idx_map[fig["objectKey"]]
    elif "id" in objects_map[0]:
        for i, m in enumerate(objects_map):
            class_to_idx_map[m["id"]] = i
        resolve_class_idx = lambda fig: class_to_idx_map[fig["objectId"]]
    else:
        raise ValueError("The JSON annotations file is expected to have either objects[...].key identifier, or objects[...].id")

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

            data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index, object_key=fig.get("objectKey",""), x1=x1, x2=x2, y1=y1, y2=y2, idx=idx)
            #data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
            datas.append(data)
            idx += 1

    return pd.DataFrame(datas)

def convert_images_annotations_folder(source: Union[str|dict], meta_file: str) -> pd.DataFrame:
    # Each DataFrame in dfs will correspond to 1 image
    dfs = [] 
    
    annotations_by_file = None
    if source is None:
        raise ValueError("source argument is mandatory")
    elif isinstance(source, str):
        if not os.path.isdir(source) or not os.path.exists(source):
            raise ValueError("If source is passed as string, it needs to represent an existing directory")
      
    elif isinstance(source, dict):
        if not isinstance(source[source.keys()[0]], dict):
            raise ValueError("If source is passed as a dict, it needs to be a dict (keyed by file name) of JSON objects")
        annotations_by_file = source
    else:
        raise ValueError("Unsupported type of source argument")
    
    meta_map = {}
    with open(meta_file, 'r') as f:
        meta_key_map = json.load(f)
    
    for i, cl in enumerate(meta_key_map["classes"]):
        meta_map[cl["id"]] = i

    for i, annotations in enumerate(annotations_by_file):
        dfs.append(convert_single_image_annotation_file(annotations, i, meta_map))
    return pd.concat(dfs, axis=0)


def convert_single_image_annotation_file(annotations: dict, frame_index: int, meta_map: dict) -> pd.DataFrame:
    # Each element in datas correspond to 1 bounding box
    datas = []
    (width, height) = annotations["size"]["width"], annotations["size"]["height"]

    resolve_class_idx = lambda fig: meta_map[fig["classId"]]
    idx = 0
    for detected in annotations["objects"]:
        class_id = resolve_class_idx(detected)
            
        (x1, y1) = detected["points"]["exterior"][0]
        (x2, y2) = detected["points"]["exterior"][1]

        box_arr = np.array([x1, y1, x2, y2], dtype='float')
        box_scaled = _xyxy2xywhn(box_arr, w=width, h=height)

        data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index, object_key="", x1=x1, x2=x2, y1=y1, y2=y2, idx=idx)
        # data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
        datas.append(data)
        idx += 1
    return pd.DataFrame(datas)

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
