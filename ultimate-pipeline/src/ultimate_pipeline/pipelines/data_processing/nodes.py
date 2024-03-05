"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

import typing as t
import json 
import logging
import os
from pathlib import Path
import supervisely as sly
import pandas as pd
from .supervisely_converter import convert_video_annotations, convert_images_annotations_folder

logger = logging.getLogger(__name__)

def download_videos_from_supervisely(parameters: t.Dict) -> t.Tuple:

    logger.info("Downloading video from Supervisely")

    videos_path = Path(parameters["video_path"])
    annotations_path = Path(parameters["annotations_path"])

    if not os.path.exists(videos_path):
        os.makedirs(videos_path)

    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    api = sly.Api.from_env()

    video_info_list = api.video.get_list(parameters["dataset_id"])

    video_name_list = []

    for video in video_info_list:

        video_id = video.id
        video_name = Path(video.name)
        save_path_video = videos_path / video_name
        save_path_annotation = str(annotations_path / video_name.stem) + ".json"
    
        #save video
        api.video.download_path(video_id, save_path_video)

        #save annotation
        video_ann_json = api.video.annotation.download(video_id)
        with open(save_path_annotation, "w") as outfile: 
            json.dump(video_ann_json, outfile, indent=4)
        
        video_name_list.append(video_name.stem)

    return video_name_list

def convert_supervisely_video_annotations_to_dataframe(json_data):
    return convert_video_annotations(json_data)

def create_yolo_frame_partitions(df: pd.DataFrame) -> t.Dict[str, t.Any]:
    """
    Split the DataFrame by 'frame' column into a dictionary of partitioned data frames 
    keyed by 'frame' (which corresponds to a video frame number or name). 
    """
    grouped = df.groupby(by="frame")
    return { str(frame): data for frame, data in grouped }

def convert_supervisely_images_annotations_to_dataframe(
        annotation_partitions: dict[str, t.Callable[[],dict]], 
        meta_file: t.Callable[[],dict]) -> pd.DataFrame:
    logger.info("Reading folder data")
    logger.info(f"annotation_partitions type: {type(annotation_partitions)}")
    logger.info(f"meta_file type: {type(meta_file)}")

    df = convert_images_annotations_folder(source=annotation_partitions, meta_file=meta_file)
    return df
    