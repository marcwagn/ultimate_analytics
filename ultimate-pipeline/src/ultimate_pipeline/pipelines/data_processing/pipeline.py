"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_image_dataset_from_supervisely, convert_supervisely_images_annotations_to_dataframe, create_yolo_dataframe_partitions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            download_image_dataset_from_supervisely,
            inputs=["params:image_dataset"],
            outputs=["supervisely_images_metadata", "supervisely_images_annotation", "supervisely_images"],
            name="download_image_dataset_from_supervisely_node",
        ),
        node(
            convert_supervisely_images_annotations_to_dataframe, 
            inputs=["supervisely_images_annotation", "supervisely_images_metadata"], 
            outputs="yolo_dataframe_variable", 
            name="convert_supervisely_images_annotations_to_dataframe_node"
        ),
        node(
            create_yolo_dataframe_partitions, 
            inputs=["yolo_dataframe_variable"], 
            outputs="yolo_annotation_txt", 
            name="split_yolo_dataframe_into_partitions_node"
        ),
    ])
