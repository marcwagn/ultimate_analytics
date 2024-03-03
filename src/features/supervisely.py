import json
import pandas as pd
import numpy as np
import os

class VideoAnnotationsConverter:
    def __init__(self, label_file: str):
        """
        Initialize a class of VideoAnnotationsConverter for conversion on Supervisely annotation format.
        References:
        - Supervisely format: https://developer.supervisely.com/getting-started/supervisely-annotation-format
        - YOLO v5 format: https://docs.ultralytics.com/datasets/detect/p
        """
        self.label_file = label_file

    def read_bounding_boxes_dataframe(self) -> pd.DataFrame:
        """
        Convert Supervisely annotations to a DataFrame containing normalized bounding boxes info in the following format:

            - cls (int) - class id
            - x (float) - bounding box centre x
            - y (float) - bounding box centre x
            - w (float) - bounding box width
            - h (float) - bounding box width
            - frame (str) - frame number or name 
        """
        with open(self.label_file, 'r') as f:
            annotations =  json.load(f)

        objects_map = annotations["objects"]

        class_key_to_idx_map = {}

        for i, m in enumerate(objects_map):
            class_key_to_idx_map[m["key"]] = i

        # Each DataFrame in dfs will correspond to 1 bounding box
        dfs = [] 
        (width, height) = annotations["size"]["width"], annotations["size"]["height"]

        for frame in annotations["frames"]:
            frame_index = frame["index"]
            for fig in frame["figures"]:
                class_id = class_key_to_idx_map[fig["objectKey"]]
            
                (x1, y1) = fig["geometry"]["points"]["exterior"][0]
                (x2, y2) = fig["geometry"]["points"]["exterior"][1]

                box_arr = np.array([x1, y1, x2, y2], dtype='float')
                box_scaled = VideoAnnotationsConverter._xyxy2xywhn(box_arr, w=width, h=height)

                data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], frame=frame_index)
                dfs.append(pd.DataFrame([data]))

        return pd.concat(dfs, axis=0)

    @staticmethod
    def convert_to_yolo_txts(df: pd.DataFrame, output_path: str):
        groups = df.groupby('frame_no')
        for g in groups:
            frame_no, df = g
            df = df[["cls", "x", "y", "w", "h"]]
            filepath = os.path.join(output_path, f"frame_{frame_no:06}.txt")
            df.to_csv(filepath, index=False, header=False, sep=' ')

    @staticmethod
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