import json
import pandas as pd
import numpy as np
from ultralytics.utils.ops import xyxy2xywhn
import os

class VideoAnnotationsConverter:
    def __init__(self, label_file: str, key_id_map_file: str):
        """
        Initialize a class of VideoAnnotationsConverter for conversion on Supervisely annotation format.
        """
        self.label_file = label_file
        self.key_id_map_file = key_id_map_file

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
        with open(self.key_id_map_file, 'r') as f:
            key_id_map =  json.load(f)
        with open(self.label_file, 'r') as f:
            labels_all =  json.load(f)

        class_key_to_id_map = key_id_map["objects"]
        figure_key_to_id_map = key_id_map["figures"]

        class_key_to_idx_map = {}
        figure_key_to_idx_map = {}

        for i, k in enumerate(class_key_to_id_map):
            class_key_to_idx_map[k] = i

        for i, k in enumerate(figure_key_to_id_map):
            figure_key_to_idx_map[k] = i

        # Each DataFrame in dfs will correspond to 1 bounding box
        dfs = [] 
        (width, height) = labels_all["size"]["width"], labels_all["size"]["height"]

        for frame in labels_all["frames"]:
            frame_index = frame["index"]
            for fig in frame["figures"]:
                figure_id = figure_key_to_idx_map[fig["key"]]
                class_id = class_key_to_idx_map[fig["objectKey"]]
            
                (x1, y1) = fig["geometry"]["points"]["exterior"][0]
                (x2, y2) = fig["geometry"]["points"]["exterior"][1]

                box_arr = np.array([x1, y1, x2, y2], dtype='float')
                box_scaled = xyxy2xywhn(box_arr, w=width, h=height)

                data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], id=figure_id, frame_no=frame_index)
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