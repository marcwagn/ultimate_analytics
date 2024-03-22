from typing import Tuple, Union
import numpy as np
import pandas as pd

candidate_clsid_pairs = np.array([[31,32],[34,35],[39,40],[41,42]])
#keypoint_lines_map = {31: "TC", 32: "TC", 34: "TF", 35: "TF", 39: "BF", 40: "BF", 41: "BC", 42: "BC"}
#keypoint_lines_beg_or_end_map = {31: 0, 32: 1, 34: 0, 35: 1, 39: 0, 40: 1, 41: 0, 42: 1}
keypoints_to_lines_map = {
    31: dict(name="TLC", line="TC", lr=0, pref=3),
    32: dict(name="TRC", line="TC", lr=1, pref=3),
    34: dict(name="TLF", line="TF", lr=0, pref=2),
    35: dict(name="TRF", line="TF", lr=1, pref=2),
    39: dict(name="BLF", line="BF", lr=0, pref=1),
    40: dict(name="BRF", line="BF", lr=1, pref=1),
    41: dict(name="BLC", line="BC", lr=0, pref=0),
    42: dict(name="BRC", line="BC", lr=1, pref=0),
}

lines_to_keypoints_map = {
    "TC": dict(cls_ids=[31,32], pref=3, real_pitch_coords=np.float32([[0,0],[0,37]])),
    "TF": dict(cls_ids=[34,35], pref=2, real_pitch_coords=np.float32([[18,0],[18,37]])),
    "BF": dict(cls_ids=[39,40], pref=1, real_pitch_coords=np.float32([[82,0],[82,37]])),
    "BC": dict(cls_ids=[41,42], pref=0, real_pitch_coords=np.float32([[100,0],[100,37]])),
}

conf_threshold = 0.6

def get_number_of_keypoint_pairs(group: pd.DataFrame) -> int:
    candidate_cls_df = _filter_and_augment_with_keypoint_columns(group)
    keypoint_counts = candidate_cls_df.groupby("keypoint_line")["cls"].aggregate("count")
    return len(keypoint_counts[keypoint_counts >= 2])

def get_ratio_of_frames_with_fewer_than_2x2_keypoints(df: pd.DataFrame, n: int=2, conf_threshold=conf_threshold) -> float:
    df_filtered = df.groupby('frame').filter(lambda g: get_number_of_keypoint_pairs(g) < n)
    return len(df_filtered['frame'].unique())/len(df['frame'].unique())

class KeypointDetectionResult:
    """
    Initialize KeypointDetectionResult.
    Args:
        keypoint_line_keys (list[str]): list of keypoint line identifiers (e.g. "TC", "BF")
        keypoints (np.ndarray): an numpy array of shape (4,2) consisting of keypoint coordinates in YOLO normalized format
        keypoints_real_pitch_coords (np.ndarray): an numpy array of shape (4,2) consisting of coordinates of the real Ultimate pitch that correspond to keypoints
    """
    def __init__(self, keypoint_line_keys: list[str], keypoints: np.ndarray):
        self.keypoint_line_keys = keypoint_line_keys
        self.keypoints = keypoints

def get_4_best_keypoint_pairs(df: pd.DataFrame, frame_no: int) -> Union[KeypointDetectionResult, None]:
    """
    Calculate 4 best keypoint pairs.
    Args:
        df (pd.DataFrame): DataFrame containing prediction data for a single image/video frame
        frame_no (int): image/video frame number
    """
    # Augment the data with additional keypoint-specific columns
    df_augmented = _filter_and_augment_with_keypoint_columns(df)
    # First, sort the keypoints by our preference (bottom-up, as those closer are more precise) and begin-or-end flag
    #df_sorted = df_augmented.sort_values(by=["keypoint_line_pref", "keypoint_line_lr"], ascending=True)
    # Count the number of keypoints in each keypoint line group
    df_agg = df_augmented.groupby("keypoint_line").agg(count=("keypoint_line", "count")).reset_index()
    df_agg["keypoint_line_pref"] = df_agg["keypoint_line"].apply(lambda c: lines_to_keypoints_map[c]["pref"])
    s_keypoints_by_count_and_pref = df_agg.sort_values(by=["count", "keypoint_line_pref"]).set_index("keypoint_line")["count"]
    #s_keypoints_by_count = df_sorted.groupby("keypoint_line").aggregate("count").sort_values(by=["keypoint_line_pref", "keypoint_line_lr"])

    # Sanity check - a situation with multiple keypoints instances of the same class may indicate a problem with the model
    if len(s_keypoints_by_count_and_pref[s_keypoints_by_count_and_pref >=3]) > 0:
        raise ValueError(f"Detected more than 3 keypoints of same type in frame {frame_no}")
    
    s_at_least_2 = s_keypoints_by_count_and_pref[s_keypoints_by_count_and_pref == 2]
    if len(s_at_least_2) < 2:
        # Not enough keypoint candidates
        return None
    keypoint_line_keys_top_2 = list(s_at_least_2.keys())[0:2]
    keypoint_coords_list = []
    for key in keypoint_line_keys_top_2:
        for clsid in lines_to_keypoints_map[key]["cls_ids"]:
            index_mask = df_augmented["cls"]==clsid
            coords = np.float32(df_augmented.loc[index_mask, "x":"y"]).squeeze()
            keypoint_coords_list.append(coords)
        #keypoints_real_pitch_coords_list.append(lines_to_keypoints_map[key]["real_pitch_coords"])

    return KeypointDetectionResult(keypoint_line_keys=keypoint_line_keys_top_2, keypoints=np.stack(keypoint_coords_list))

def _filter_and_augment_with_keypoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe, leaving only the keypoints on the longer edge of the pitch,
    and augment the dataframe with additional columns.
    Additional columns:
        keypoint_line - one of four pre-defined lines on the pitch (all parallel to each other)
        keypoint_line_lr - whether the keypoint is on the left or right hand side of the pitch (from camera POV)
        keypoint_line_pref - algorithm's preference of choosing a line (used as tie-breaker)
    """
    mask = (df["cls"].isin(candidate_clsid_pairs.flatten())) & (df["conf"] > conf_threshold)
    candidate_cls_df = df[mask].copy(deep=False)
    candidate_cls_df["keypoint_line"] = candidate_cls_df["cls"].apply(lambda c: keypoints_to_lines_map[c]["line"])
    candidate_cls_df["keypoint_line_lr"] = candidate_cls_df["cls"].apply(lambda c: keypoints_to_lines_map[c]["lr"])
    candidate_cls_df["keypoint_line_pref"] = candidate_cls_df["cls"].apply(lambda c: keypoints_to_lines_map[c]["pref"])
    return candidate_cls_df
