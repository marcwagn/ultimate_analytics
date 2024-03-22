import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from src.keypoints_helper import get_4_best_keypoint_pairs

def _read_predictions_and_filter_keypoints(file_path: Path, is_tracking: bool=True, frame_no: int=0):
    columns=["cls", "x", "y", "w", "h", "conf", "id"] if is_tracking else ["cls", "x", "y", "w", "h", "conf"]
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    df["frame"] = frame_no
    df["filename"] = file_path.stem
    keypoints_df = df[df["cls"] >= 31]
    return keypoints_df

def test_get_4_best_keypoint_pairs_choose_TF_TC():
    prefix = "./src/tests/data/tracking_set_1"
    file_path = Path(prefix)/"pony_vs_the_killjoys_pool_004_1107.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True)

    result = get_4_best_keypoint_pairs(df, frame_no=1107)
    assert result is not None
    assert result.keypoint_line_keys is not None
    print(result.keypoint_line_keys)

    assert result.keypoint_line_keys == ["TF", "TC"]
    assert result.keypoints is not None
    assert isinstance(result.keypoints, np.ndarray)
    assert result.keypoints.shape == (4,2)
    expected_keypoints = np.float32([
        [0.152733, 0.375162],
        [0.825292, 0.372376],
        [0.230659, 0.288993],
        [0.753578, 0.285994]])
    assert_array_equal(result.keypoints, expected_keypoints)

    assert result.cls_ids is not None
    assert result.cls_ids == [34,35,31,32]

def test_get_4_best_keypoint_pairs_choose_BF_TF():
    prefix = "./src/tests/data/tracking_set_1"
    file_path = Path(prefix)/"pony_vs_the_killjoys_pool_004_2191.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True)

    result = get_4_best_keypoint_pairs(df, frame_no=2191)
    assert result is not None
    assert result.keypoint_line_keys is not None
    print(result.keypoint_line_keys)

    assert result.keypoint_line_keys == ["BF", "TF"]

    assert result.keypoints is not None
    assert isinstance(result.keypoints, np.ndarray)
    assert result.keypoints.shape == (4,2)
    expected_keypoints = np.float32([
        [0.0338198, 0.615392],
        [0.933916, 0.615022],
        [0.323212, 0.236336],
        [0.734717, 0.235054]])
    assert_array_equal(result.keypoints, expected_keypoints)

    assert result.cls_ids is not None
    assert result.cls_ids == [39,40,34,35]

def test_get_4_best_keypoint_pairs_not_enough_keypoints_and_no_fallback_possible():
    prefix = "./src/tests/data/tracking_set_1"
    file_path = Path(prefix)/"pony_vs_the_killjoys_pool_004_1_not_enough_kp.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True)

    result = get_4_best_keypoint_pairs(df, frame_no=1)
    assert result is None

def test_get_4_best_keypoint_pairs_duplicate_keypoints():
    prefix = "./src/tests/data/tracking_set_1"
    file_path = Path(prefix)/"pony_vs_the_killjoys_pool_004_1_duplicate.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True)

    with pytest.raises(ValueError, match="Detected more than 3 keypoints of same type in frame 1"):
        get_4_best_keypoint_pairs(df, frame_no=1)
  