import json
import pandas as pd
import numpy as np
from typing import Callable, Union, TypeAlias
import os
import logging

JSONGenerator: TypeAlias = Callable[[],dict]
# Python 3.12: type JSONGenerator = Callable[[],dict]

logger = logging.getLogger(__name__)

def convert_video_annotations(source: Union[str|dict]) -> pd.DataFrame:
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

    class_to_idx_map = { m[objects_class_id_key_name]: i for i, m in enumerate(objects_list)}
    resolve_class_idx = lambda fig: class_to_idx_map[fig[figures_class_id_key_name]]

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

def convert_images_annotations_folder(source: Union[str|dict[JSONGenerator]], meta_file: Union[str|dict]) -> pd.DataFrame:
    # Each DataFrame in dfs will correspond to 1 image
    dfs = [] 
    
    # kkk = next(iter(source))
    # logger.info(f"This is what was passed: {type(kkk)}")
    # logger.info(f"This is what was passed: {kkk}")
    # logger.info(f"This is what was passed: {source[kkk]}")
    # truthy = [isinstance(o, Callable) for o in source.values()]
    # logger.info(truthy)

    annotations_generators = None
    #annotations_by_file = None
    if source is None:
        raise ValueError("source argument is mandatory")
    elif isinstance(source, str):
        if not os.path.isdir(source) or not os.path.exists(source):
            raise ValueError("If source is passed as string, it needs to represent an existing directory")
        annotations_generators = {
            file: (lambda : _read_json_content(source, file))
                for file in os.listdir(source) 
                if os.path.isfile(os.path.join(source, file))}
    elif isinstance(source, dict):
        if not all([isinstance(o, Callable) for o in source.values()]):
            raise ValueError("If source is passed as a dict, it needs to be a dict (keyed by file name) of callables returning JSON dicts")
        #annotations_by_file = source()
        annotations_generators = source
    else:
        raise ValueError("Unsupported type of source argument")
    
    # if meta_file is None:
    #     raise ValueError("meta_file argument is mandatory")
    # if not isinstance(meta_file, str) and not isinstance(meta_file, JSONGenerator):
    #     raise ValueError("meta_file needs to be a str or a callable returning a JSON dict")
    
    meta_file_content = None
    if isinstance(meta_file, str):
        with open(meta_file, 'r') as f:
            meta_file_content = json.load(f)
    elif isinstance(meta_file, dict):
        meta_file_content = meta_file
    else:
        raise ValueError("meta_file needs to be a str or a JSON dict")
            
    meta_map = { m["id"]: i for i, m in enumerate(meta_file_content["classes"]) }


    for key, annotations_gen in annotations_generators.items():
        dfs.append(convert_single_image_annotation_file(annotations_gen(), key, meta_map))
    return pd.concat(dfs, axis=0)


def convert_single_image_annotation_file(annotations: dict, frame_key: str, meta_map: dict) -> pd.DataFrame:
    """
    Convert a Supervisely annotation for a single image to normalized bounding boxes.
    Args:
        annotations (dict): content of JSON annotation file
    """
    # Each element in datas correspond to 1 bounding box
    datas = []
    (width, height) = annotations["size"]["width"], annotations["size"]["height"]

    resolve_class_idx = lambda fig: meta_map[fig["classId"]]
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

def _read_json_content(root, filename):
    with open(os.path.join(root, filename)) as f:
        return json.load(f)