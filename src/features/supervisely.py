import json
import pandas as pd
import numpy as np
from ultralytics.utils.ops import xyxy2xywhn
import os

def convert_to_yolo_df(key_id_map_file: str, label_file: str) -> pd.DataFrame:
    with open(key_id_map_file, 'r') as f:
        key_id_map =  json.load(f)
    with open(label_file, 'r') as f:
        labels_all =  json.load(f)

    class_key_to_id_map = key_id_map["objects"]
    figure_key_to_id_map = key_id_map["figures"]

    class_key_to_idx_map = {}
    figure_key_to_idx_map = {}

    for i, k in enumerate(class_key_to_id_map):
        class_key_to_idx_map[k] = i

    for i, k in enumerate(figure_key_to_id_map):
        figure_key_to_idx_map[k] = i

    #NB - since Python 3.7, iteration of dict matches the order of insertion
    # classes = dict()
    # for class_def in labels_all["objects"]:
    #     classes[class_def["key"]] = class_def["classTitle"]
    
    dfs = [] # Each pd.DataFrame will correspond to 1 bounding box
    (width, height) = labels_all["size"]["width"], labels_all["size"]["height"]
    # class	x	y	w	h	id	
    for frame in labels_all["frames"]:
        frame_index = frame["index"]
        for fig in frame["figures"]:
            figure_id = figure_key_to_idx_map[fig["key"]]
            class_id = class_key_to_idx_map[fig["objectKey"]]
           
            (x1, y1) = fig["geometry"]["points"]["exterior"][0]
            (x2, y2) = fig["geometry"]["points"]["exterior"][1]
            box_arr = np.array([x1, y1, x2, y2], dtype='float')

            # x = box_arr
            # w = width
            # h = height
            # y = np.empty_like(x, dtype='float')  # faster than clone/copy
            # y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
            # y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
            # y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
            # y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height

            # bbb = ((box_arr[..., 0] + box_arr[..., 2]) / 2) / width
            # y[..., 0] = bbb
            box_scaled = xyxy2xywhn(box_arr, w=width, h=height)
            # (w, h) = (x2-x1, y2-y1)
            data = dict(cls=class_id, x=box_scaled[0], y=box_scaled[1], w=box_scaled[2], h=box_scaled[3], id=figure_id, frame_no=frame_index)
            #mini_df = pd.DataFrame([data], index=[figure_id, frame_index])
            dfs.append(pd.DataFrame([data]))
            #output_per_frame.append((class_id, box_scaled[0], box_scaled[1], box_scaled[2], box_scaled[3], figure_id))

    return pd.concat(dfs, axis=0)

def dump_as_yolo_txt(df: pd.DataFrame, output_path: str):
    #if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

    groups = df.groupby('frame_no')
    for g in groups:
        frame_no, df = g
        df = df[["cls", "x", "y", "w", "h"]]
        filepath = os.path.join(output_path, f"frame_{frame_no:06}.txt")
        df.to_csv(filepath, index=False, header=False, sep=' ')


                #https://github.com/Delilovic/supervisely_yolo/blob/master/supervisely_yolo_multiple.py
                # points = obj["points"]["exterior"]
                # w = json_object["size"]["width"]
                # h = json_object["size"]["height"]
                # w_point = points[1][0] - points[0][0]
                # h_point = points[1][1] - points[0][1]
                # x1 = round((points[0][0] + w_point / 2) / w, 5)
                # y1 = round((points[0][1] + h_point / 2) / h, 5)
                # x2 = round(w_point / w, 5)
                # y2 = round(h_point / h, 5)

    #key_id_map = json.loads(key_id_map_file)
    #key_id_map["figures"]

    #object_map = pd.DataFrame.normalize_json(labels_all, record_path="objects")
            
        #labels_all["objects"]
    # "key": class id (object_key) (e052e3e5ae0c474fa01dffa588f4856f)
    # "classTitle"

    # dict key is the figure_id
    # dict value is the class id

    #see also https://github.com/Delilovic/supervisely_yolo

    #  "objects": [
    #     {
    #         "id": 1219977368,
    #         "classId": 12217172,
    #         "datasetId": 918584,
    #         "labelerLogin": "adam_jasinski",
    #         "createdAt": "2024-02-10T13:07:13.637Z",
    #         "updatedAt": "2024-02-10T13:07:13.637Z",
    #         "tags": [],
    #         "entityId": 334517698,
    #         "classTitle": "Player"
    #     },
    #     {
    #         "id": 1219987266,
    #         "classId": 12341909,
    #         "datasetId": 918584,
    #         "labelerLogin": "adam_jasinski",
    #         "createdAt": "2024-02-18T14:31:22.000Z",
    #         "updatedAt": "2024-02-18T14:31:22.000Z",
    #         "tags": [],
    #         "entityId": 334517698,
    #         "classTitle": "Frisbee"
    #     }
    # ],