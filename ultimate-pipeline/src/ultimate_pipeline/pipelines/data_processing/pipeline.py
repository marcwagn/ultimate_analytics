"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (download_image_dataset_from_supervisely, 
                    convert_supervisely_annotations_to_yolo_detect_dataframe, 
                    convert_supervisely_annotations_to_yolo_pose_dataframe,
                    partition_dataframe_into_dict, train_val_split)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            download_image_dataset_from_supervisely,
            inputs=["params:image_dataset"],
            outputs=["supervisely_metadata", "supervisely_annotation", "supervisely_images"],
            name="download_image_dataset_from_supervisely_node",
        ),
        node(
            convert_supervisely_annotations_to_yolo_detect_dataframe, 
            inputs=["supervisely_annotation", "supervisely_metadata"], 
            outputs="yolo_detect_annotation_dataframe",
            name="convert_supervisely_annotations_to_yolo_detect_dataframe_node"
        ),
        node(
            partition_dataframe_into_dict, 
            inputs=["yolo_detect_annotation_dataframe"], 
            outputs="yolo_detect_annotation", 
            name="create_yolo_detect_annotation_node"
        ),
        node(
            train_val_split,
            inputs=["params:ultimate_object_detect",
                    "supervisely_images",
                    "yolo_detect_annotation"],
            outputs=["yolo_detect_images_train",
                     "yolo_detect_images_val",
                     "yolo_detect_annotation_train",
                     "yolo_detect_annotation_val"],
            name="train_val_split_detect_node",
        ),
        node(
            convert_supervisely_annotations_to_yolo_pose_dataframe, 
            inputs=["supervisely_annotation", "supervisely_metadata"], 
            outputs="yolo_pose_annotation_dataframe",
            name="convert_supervisely_annotations_to_yolo_pose_dataframe_node"
        ),
        node(
            partition_dataframe_into_dict, 
            inputs=["yolo_pose_annotation_dataframe"], 
            outputs="yolo_keypoints_annotation_train", 
            name="create_partitioned_dataframe_pose_node"
        ),
    ])
