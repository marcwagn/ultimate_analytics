"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_videos_from_supervisely, convert_supervisely_to_dataframe

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            download_videos_from_supervisely,
            inputs=["params:video_dataset"],
            outputs=["video_name_list"],
            name="download_videos_from_supervisely_node",
        ),
        node(
            convert_supervisely_to_dataframe,
            inputs=["supervisely_video_annotations"],
            outputs="yolo_annotations",
            name="convert_supervisely_to_dataframe_node",
        ),
    ])
