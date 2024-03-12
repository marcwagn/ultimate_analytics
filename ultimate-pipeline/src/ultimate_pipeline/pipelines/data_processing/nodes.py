"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""
from typing import Callable, Any
import pandas as pd

import logging
import random

from .supervisely_converter import convert_images_annotations_folder
from .supervisely_downloader import helper_download_image_dataset_from_supervisely

logger = logging.getLogger(__name__)

def download_image_dataset_from_supervisely(params: dict[str, Any]) -> dict[str, Any]:
    
    logger.info("Downloading dataset from Supervisely")

    return helper_download_image_dataset_from_supervisely(params)


def partion_dataframe_into_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the DataFrame by 'frame' column into a dictionary of partitioned data frames 
    keyed by 'frame' (which corresponds to a video frame number or name). 
    """
    grouped = df.groupby(by="frame")
    return { str(frame): data for frame, data in grouped }

def convert_supervisely_annotations_to_yolo_format_dataframe(
        annotation_partitions: dict[str, Callable[[], dict[str, Any]]], 
        meta_file: dict) -> pd.DataFrame:
    """
    Convert Supervisely image annotations from a given folder to a DataFrame containing bounding boxes.
    Args:
        annotation_partitions (dict) - a dictionary (keyed by file names) containing content generator functions (Callables generating JSON annotation content as dict)
        meta_file (dict) - JSON content of meta.json
    """
    logger.info("Reading folder data")
    annotation_partitions_without_extension = {}
    for filename, content_generator in annotation_partitions.items():
        filename_without_extension = filename[:filename.rfind(".")]
        annotation_partitions_without_extension[filename_without_extension] = content_generator

    return convert_images_annotations_folder(source=annotation_partitions_without_extension, meta_file=meta_file)

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


#
#def copy_dataset_items(items: dict[str, Any]) -> dict[str, Any]:
#    """Copy data between datasets"""
#    return items
#
#    