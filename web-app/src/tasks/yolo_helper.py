import ultralytics
import pandas as pd
import numpy as np

from .team_detector import TeamDetector

frames = 0

def make_callback_adapter_with_counter(event_name, callback):
    """
    Convert the callback function with 2 params to a callback format required by YOLO.
    Args:
        event_name: str: YOLO pipeline event name
        callback: Callable(str, int): a callback function accepting an 2 params: event_name and counter
    Return:
        A callback in the format required by YOLO.
    """
    event_counter = 0

    def yolo_callback(component):
        nonlocal event_counter
        event_counter += 1
        callback(event_name, event_counter)

    return yolo_callback

def _convert_single_tracking_result(frame_no, boxes_result:ultralytics.engine.results.Boxes) -> pd.DataFrame:
    box = boxes_result.boxes # sic!
    int_vectorized = np.vectorize(np.int_, otypes=[int])
    if box is not None:
        # NB - Torch tensors may sit on GPU if one was used, they need to be moved to CPU to convert to Numpy
        class_ids = int_vectorized(box.cls.cpu().numpy())
        observation_count = len(class_ids)

        def class_id_to_name(id):
            return boxes_result.names[int(id)]

        class_names = list(map(class_id_to_name, class_ids))
        ids = int_vectorized(box.id.cpu()) if box.id is not None else np.zeros(shape=observation_count, dtype='int')
        # NB - use the normalized bounded boxes
        xywhn = box.xywhn.cpu() 
        xs = xywhn[:, 0]
        ys = xywhn[:, 1]
        ws = xywhn[:, 2]
        hs = xywhn[:, 3]
        confs = box.conf.cpu()
        data = dict(cls=class_ids, x=xs, y=ys, w=ws, h=hs, conf=confs, id=ids, frame=frame_no, cls_name=class_names)
        df = pd.DataFrame(data=data)
        return df
    else:
        return pd.DataFrame(columns=['cls', 'x', 'y', 'w', 'h', 'conf', 'id', 'frame', 'cls_name'])

def convert_tracking_results_to_pandas(tracking_results: list[ultralytics.engine.results.Results]) -> pd.DataFrame:
    """
    Convert YOLOv8 tracking output to a Pandas DataFrame.
    The DataFrame contains the following columns:
        - frame (int): frame number
        - cls (int) - class identifier
        - cls_name (str) - class name of the tracked object
        - conf (float) - class detection confidence
        - id (int): identifier of the tracked object
        - x (int) - coordinates of the bounding boxes
        - y (int)
        - w (int)
        - h (int)
        - frame (int) - frame number
    Args:
        tracking_results (list[ultralytics.engine.results.Results]) - YOLO tracking results
    """
    dfs = [] # Will contain 1 data frame per video frame
    for i, tr in enumerate(tracking_results):
        df = _convert_single_tracking_result(i, tr)
        
        # init team prediction for non players
        df["team"] = -1 
        # update team prediction for players
        pred_teams_df = _get_team_prediction(tr)
        for _, row in pred_teams_df.iterrows():
            df.loc[df['id'] == row['id'], 'team'] = row['pred_team']

        dfs.append(df)

    return pd.concat(dfs)

def _get_team_prediction(tracking_results: list[ultralytics.engine.results.Results]) -> pd.DataFrame:
    """
    Get the team prediction from the tracking results.
    Args:
        tracking_results (ultralytics.engine.results.Results): The tracking results.
    Returns:
        pd.DataFrame: The team prediction.
    """
    frame = tracking_results.orig_img
    detector = TeamDetector(frame, tracking_results)
    lst_player_imgs = detector.get_player_images()

    if not lst_player_imgs:
        return pd.DataFrame(columns=["id", "pred_team"])

    return detector.predict_player_clusters(lst_player_imgs)