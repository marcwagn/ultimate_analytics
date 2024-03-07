"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

import typing as t
import pandas as pd

import logging

from .supervisely_converter import convert_images_annotations_folder
from .supervisely_downloader import helper_download_image_dataset_from_supervisely

logger = logging.getLogger(__name__)

def download_image_dataset_from_supervisely(params: t.Dict) -> t.Dict[str, t.Any]:
    
    logger.info("Downloading dataset from Supervisely")

    return helper_download_image_dataset_from_supervisely(params)


def create_yolo_dataframe_partitions(df: pd.DataFrame) -> t.Dict[str, pd.DataFrame]:
    """
    Split the DataFrame by 'frame' column into a dictionary of partitioned data frames 
    keyed by 'frame' (which corresponds to a video frame number or name). 
    """
    grouped = df.groupby(by="frame")
    return { str(frame): data for frame, data in grouped }

def convert_supervisely_images_annotations_to_dataframe(
        annotation_partitions: dict[str, t.Callable[[],dict]], 
        meta_file: dict) -> pd.DataFrame:
    """
    Convert Supervisely image annotations from a given folder to a DataFrame containing bounding boxes.
    Args:
        annotation_partitions (dict) - a dictionary (keyed by file names) containing content generator functions (Callables generating JSON annotation content as dict)
        meta_file (dict) - JSON content of meta.json
    """
    logger.info("Reading folder data")
    logger.info(f"annotation_partitions type: {type(annotation_partitions)}")
    logger.info(f"meta_file type: {type(meta_file)}")

    return convert_images_annotations_folder(source=annotation_partitions, meta_file=meta_file)
    