"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

import typing as t
import pandas as pd

import logging

from .supervisely_converter import convert_to_normalized_bounding_boxes
from .supervisely_downloader import helper_download_image_dataset_from_supervisely

logger = logging.getLogger(__name__)

def download_image_dataset_from_supervisely(params: t.Dict) -> t.Dict[str, t.Any]:
    
    logger.info("Downloading dataset from Supervisely")

    return helper_download_image_dataset_from_supervisely(params)
  

def convert_supervisely_to_dataframe(json_data):
    return convert_to_normalized_bounding_boxes(json_data)

def create_yolo_frame_partitions(df: pd.DataFrame) -> t.Dict[str, t.Any]:
    grouped = df.groupby(by="frame")

    frame_dict = {}
    for frame, data in grouped:
        frame_dict[str(frame)] = data

    return frame_dict