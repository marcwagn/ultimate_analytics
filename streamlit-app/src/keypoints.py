from typing import Union
import numpy as np
import pandas as pd

class KeypointQuad:
    """
    Contains information about 4 keypoints forming a quadrangle.
    Args:
        keypoint_line_keys (list[str]): list of keypoint line identifiers (e.g. "TC", "BF")
        keypoints (np.ndarray): an numpy array of shape (4,2) consisting of keypoint coordinates in YOLO normalized format
        cls_ids (list[int]): a list of class ids identifying specific keypoints (length=4)
    """
    def __init__(self, keypoint_line_keys: list[str], keypoints: np.ndarray, cls_ids: list[int]):
        self._keypoint_line_keys = keypoint_line_keys
        self._keypoints = keypoints
        self._cls_ids = cls_ids

    """Keypoint line identifiers (e.g. "TC", "BF")"""
    @property
    def keypoint_line_keys(self):
        return self._keypoint_line_keys
    
    """Keypoints as numpy array of shape (4,2)"""
    @property
    def keypoints(self):
        return self._keypoints
    
    @property
    def cls_ids(self):
        return self._cls_ids

class KeypointsExtractor:
    """
    Initialize KeypointsExtractor.
    Args:
        all_frames_df (pd.DataFrame): DataFrame with keypoint predictions for all video frames/images
        conf_threshold (float): confidence threshold to recognize a keypoint as valid
        max_lookback (int): how many video frames/images to look back if we don't have enough keypoints in current frame
    """
    def __init__(self, all_frames_df: pd.DataFrame, conf_threshold:float=0.6, max_lookback:int=15):
        self._conf_threshold = conf_threshold
        self._max_lookback = max_lookback
        self._all_frames_augmented_df = self._filter_and_augment_with_keypoint_columns(all_frames_df)

    @property 
    def conf_threshold(self):
        return self._conf_threshold
    
    @property 
    def max_lookback(self):
        return self._max_lookback

    _keypoints_to_lines_map = {
        31: dict(name="TLC", line="TC", lr=0, pref=3),
        32: dict(name="TRC", line="TC", lr=1, pref=3),
        34: dict(name="TLF", line="TF", lr=0, pref=2),
        35: dict(name="TRF", line="TF", lr=1, pref=2),
        39: dict(name="BLF", line="BF", lr=0, pref=1),
        40: dict(name="BRF", line="BF", lr=1, pref=1),
        41: dict(name="BLC", line="BC", lr=0, pref=0),
        42: dict(name="BRC", line="BC", lr=1, pref=0),
    }

    _lines_to_keypoints_map = {
        "TC": dict(cls_ids=[31,32], pref=3),
        "TF": dict(cls_ids=[34,35], pref=2),
        "BF": dict(cls_ids=[39,40], pref=1),
        "BC": dict(cls_ids=[41,42], pref=0),
    }

    _candidate_clsid_pairs = np.array([line["cls_ids"] for _, line in _lines_to_keypoints_map.items()])

    # def _get_keypoints_from_previous(self, requested_cls_ids: list[int], frame_no: int) -> np.ndarray:
    #     # Assumption: df is already augmented
    #     df = self._all_frames_df
    #     filtered_df = df[(df["cls"].isin(requested_cls_ids)) & ((frame_no - self.max_lookback) < df["frame"] < frame_no) & (df["conf"] > self.conf_threshold)]
    #     #filtered_df

    def get_4_best_keypoint_pairs(self, frame_no: int) -> Union[KeypointQuad, None]:
        """
        Calculate 4 best keypoint pairs.
        Args:
            df (pd.DataFrame): DataFrame containing prediction data for a single image/video frame
            frame_no (int): image/video frame number
        """
        # Augment the data with additional keypoint-specific columns
        df_augmented = self._all_frames_augmented_df[self._all_frames_augmented_df["frame"]==frame_no]
        # Count the number of keypoints in each keypoint line group
        df_agg = df_augmented.groupby("keypoint_line").agg(count=("keypoint_line", "count")).reset_index()
        df_agg["keypoint_line_pref"] = df_agg["keypoint_line"].apply(lambda c: self._lines_to_keypoints_map[c]["pref"])
        # Sort the keypoint lines by number of keypoints present and preference
        s_keypoints_by_count_and_pref = df_agg.sort_values(by=["count", "keypoint_line_pref"]).set_index("keypoint_line")["count"]
        
        # Sanity check - a situation with multiple keypoints instances of the same class may indicate a problem with the model
        if len(s_keypoints_by_count_and_pref[s_keypoints_by_count_and_pref >=3]) > 0:
            raise ValueError(f"Detected more than 3 keypoints of same type in frame {frame_no}")
        
        s_at_least_2 = s_keypoints_by_count_and_pref[s_keypoints_by_count_and_pref == 2]
        if len(s_at_least_2) < 2:
            # Not enough keypoint candidates
            # Found 0 eligible lines - run the whole algorith on the previous dataframe
            # Found 1 eligible line - find 1 or 2 missing keypoints in previous dataframes
            return None
        keypoint_line_keys_top_2 = list(s_at_least_2.keys())[0:2]
        keypoint_coords_list = []
        keypoint_cls_ids = []
        for key in keypoint_line_keys_top_2:
            for clsid in self._lines_to_keypoints_map[key]["cls_ids"]:
                index_mask = df_augmented["cls"]==clsid
                coords = np.float32(df_augmented.loc[index_mask, "x":"y"]).squeeze()
                keypoint_coords_list.append(coords)
                keypoint_cls_ids.append(clsid)

        return KeypointQuad(
            keypoint_line_keys=keypoint_line_keys_top_2, 
            keypoints=np.stack(keypoint_coords_list),
            cls_ids=keypoint_cls_ids)

    def _filter_and_augment_with_keypoint_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataframe, leaving only the keypoints on the longer edge of the pitch,
        and augment the dataframe with additional columns.
        Additional columns:
            keypoint_line - one of four pre-defined lines on the pitch (all parallel to each other)
            keypoint_line_lr - whether the keypoint is on the left or right hand side of the pitch (from camera POV)
            keypoint_line_pref - algorithm's preference of choosing a line (used as tie-breaker)
        """
        mask = (df["cls"].isin(self._candidate_clsid_pairs.flatten())) & (df["conf"] > self.conf_threshold)
        candidate_cls_df = df[mask].copy(deep=False)
        candidate_cls_df["keypoint_line"] = candidate_cls_df["cls"].apply(lambda c: self._keypoints_to_lines_map[c]["line"])
        candidate_cls_df["keypoint_line_lr"] = candidate_cls_df["cls"].apply(lambda c: self._keypoints_to_lines_map[c]["lr"])
        candidate_cls_df["keypoint_line_pref"] = candidate_cls_df["cls"].apply(lambda c: self._keypoints_to_lines_map[c]["pref"])
        return candidate_cls_df
