"""
Kedro nodes for 'data_processing' pipeline.
"""
from typing import Callable, Any
import pandas as pd

import logging
import random

from .supervisely_converter import convert_images_annotations_folder_to_detect_data, convert_images_annotations_folder_to_pose_data, KeypointsBoxesGenerationSettings
from .supervisely_downloader import helper_download_image_dataset_from_supervisely

logger = logging.getLogger(__name__)

def download_image_dataset_from_supervisely(params: dict[str, Any]) -> dict[str, Any]:
    
    logger.info("Downloading dataset from Supervisely")

    return helper_download_image_dataset_from_supervisely(params)


def partition_dataframe_into_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the DataFrame by 'frame' column into a dictionary of partitioned data frames 
    keyed by 'frame' (which corresponds to a video frame number or name). 
    Note: The 'frame' column is removed from the output DataFrame.
    """
    grouped = df.groupby(by="frame")
    def remove_frame_column(data_frame: pd.DataFrame) -> pd.DataFrame:
        del data_frame["frame"]
        return data_frame

    return { str(frame): remove_frame_column(data) for frame, data in grouped }

def convert_supervisely_annotations_to_yolo_detect_dataframe(
        annotation_partitions: dict[str, Callable[[], dict[str, Any]]], 
        meta_file: dict) -> pd.DataFrame:
    """
    Convert Supervisely image annotations from a given folder to a DataFrame containing bounding boxes.
    Args:
        annotation_partitions (dict) - a dictionary (keyed by file names) containing content generator functions (Callables generating JSON annotation content as dict)
        meta_file (dict) - JSON content of meta.json
    """
    logger.info("Reading supervisely annotations folder data for detect data extraction")
    annotation_partitions_without_extension = {}
    for filename, content_generator in annotation_partitions.items():
        filename_without_extension = filename[:filename.rfind(".")]
        annotation_partitions_without_extension[filename_without_extension] = content_generator

    # Arbitrary padding for bounding box generation around keypoints (graphs in Supervisely annotations)
    small_padding = (20.0, 20.0)
    dynamic_padding = {
        "31": small_padding,
        "32": small_padding,
        "33": small_padding,
        "34": small_padding,
        "35": small_padding,
        "36": small_padding,
        "37": small_padding,
        "38": (25, 25),
        "39": (30, 30),
        "40": (30, 30),
        "41": (40, 40),
        "42": (40, 40),
        "43": (40, 40),
    }
   
    keypoints_bboxes_settings = KeypointsBoxesGenerationSettings(first_keypoint_class_id=31, settings=dynamic_padding)

    yolo_detect_df = convert_images_annotations_folder_to_detect_data(
        source=annotation_partitions_without_extension, 
        meta_file=meta_file,
        keypoints_bboxes_settings=keypoints_bboxes_settings)

    # A fix up certain class ids to align with COCO class ids from the pre-trained model
    # - frisbee: 29
    # - referee: 30 (overriding existing COCO class id)
    # (see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
    fixup = {1: 29, 3: 30}
    yolo_detect_df['cls'] = yolo_detect_df['cls'].apply(lambda c: fixup.get(c, c))
    
    columns_to_include = ["cls", "x", "y", "w", "h", "frame"]
    return yolo_detect_df[columns_to_include]

def convert_supervisely_annotations_to_yolo_pose_dataframe(
        annotation_partitions: dict[str, Callable[[], dict[str, Any]]], 
        meta_file: dict) -> pd.DataFrame:
    """
    Convert Supervisely image annotations from a given folder to a DataFrame containing pose/keypoint estimation information.
    Args:
        annotation_partitions (dict) - a dictionary (keyed by file names) containing content generator functions (Callables generating JSON annotation content as dict)
        meta_file (dict) - JSON content of meta.json
    """
    logger.info("Reading supervisely annotations folder data for pose/keypoint extraction")
    annotation_partitions_without_extension = {}
    for filename, content_generator in annotation_partitions.items():
        filename_without_extension = filename[:filename.rfind(".")]
        annotation_partitions_without_extension[filename_without_extension] = content_generator

    return convert_images_annotations_folder_to_pose_data(source=annotation_partitions_without_extension, meta_file=meta_file)

def train_val_split(
        params: dict[str, Any],
        images: dict[str, Callable[[], dict[str, Any]]],
        annotations: dict[str, pd.DataFrame]):
    """ Split the dataset into training and validation sets
        Args:
            params (dict) - parameters
            images (dict) - a dictionary (keyed by file names) containing function to load image data
            annotations (dict) - a dictionary (keyed by file names) containing YOLO annotations as a DataFrame
    """

    logger.info("Splitting the dataset into training and validation sets")

    if not 0.0 < params["train_split"] <= 1.0:
        raise ValueError("train_val_split must be a float between 0 and 1")
    if params["seed"] is not None:
        random.seed(params["seed"])
    
    # shuffle data randomly or using given seed 
    files_shuffled = list(images.keys())
    random.shuffle(files_shuffled)

    split_index = int(len(files_shuffled) * params["train_split"])
    train_files = files_shuffled[:split_index]
    val_files = files_shuffled[split_index:]

    train_images = {k: images[k] for k in train_files}
    val_images = {k: images[k] for k in val_files}

    train_annotations = {k: annotations[k] for k in train_files}
    val_annotations = {k: annotations[k] for k in val_files}

    return train_images, val_images,  train_annotations, val_annotations
