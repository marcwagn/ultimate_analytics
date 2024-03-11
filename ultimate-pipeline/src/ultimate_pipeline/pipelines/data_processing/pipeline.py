"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_image_dataset_from_supervisely, convert_supervisely_images_annotations_to_dataframe, create_yolo_dataframe_partitions, copy_dataset_items

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            download_image_dataset_from_supervisely,
            inputs=["params:image_dataset"],
            outputs=["supervisely_metadata", "supervisely_annotation", "supervisely_images"],
            name="download_image_dataset_from_supervisely_node",
        ),
        node(
            convert_supervisely_images_annotations_to_dataframe, 
            inputs=["supervisely_annotation", "supervisely_metadata"], 
            outputs="yolo_dataframe_variable", 
            name="convert_supervisely_images_annotations_to_dataframe_node"
        ),
        node(
            create_yolo_dataframe_partitions, 
            inputs=["yolo_dataframe_variable"], 
            outputs="yolo_detect_annotation", 
            name="split_yolo_dataframe_into_partitions_node"
        ),
        node(
            copy_dataset_items, 
            inputs=["supervisely_images"], 
            outputs="yolo_detect_images", 
            name="copy_images_to_processed_folder_node"
        ),
    ])
