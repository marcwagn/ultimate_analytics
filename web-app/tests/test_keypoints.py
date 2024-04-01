from pathlib import Path
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from src.tasks.keypoints import KeypointsExtractor

PATH_PREFIX = Path("./tests/data/tracking_set_curated_txt")

def _read_predictions_and_filter_keypoints(file_path: Path, is_tracking: bool=True, frame_no: int=0):
    columns=["cls", "x", "y", "w", "h", "conf", "id"] if is_tracking else ["cls", "x", "y", "w", "h", "conf"]
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    df["frame"] = frame_no
    df["filename"] = file_path.stem
    return df

def test_get_4_best_keypoint_pairs_choose_TF_TC():
    file_path = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_1107.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True, frame_no=1107)

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(frame_no=1107)
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
    file_path = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_2191.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True, frame_no=2191)

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(frame_no=2191)
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
    file_path = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_1_not_enough_kp.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True, frame_no=1)

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(frame_no=1)
    assert result is None

def test_get_4_best_keypoint_pairs_duplicate_keypoints():
    file_path = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_108_duplicate.txt"
    df = _read_predictions_and_filter_keypoints(file_path, is_tracking=True, frame_no=108)

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(108)
    # Should drop the duplicate
    assert result is not None

def test_get_4_best_keypoint_pairs_only_one_keypoint_but_fallback_to_previous_frames_possible():
    file_path296 = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_296_only_one_kp.txt"
    df296 = _read_predictions_and_filter_keypoints(file_path296, is_tracking=True, frame_no=296)
    file_path295 = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_295_not_enough_with_fallback.txt"
    df295 = _read_predictions_and_filter_keypoints(file_path295, is_tracking=True, frame_no=295)
    file_path294 = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_294.txt"
    df294 = _read_predictions_and_filter_keypoints(file_path294, is_tracking=True, frame_no=294)
    df = pd.concat([df294, df295, df296])

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(frame_no=296)
    assert result is not None

    # Should fall back all the way to frame 294
    assert result.keypoint_line_keys == ["TF", "TC"]
    assert result.keypoints is not None
    assert isinstance(result.keypoints, np.ndarray)
    assert result.keypoints.shape == (4,2)
    assert result.cls_ids is not None
    assert result.cls_ids == [34,35,31,32]

    expected_keypoints = np.float32([
        [0.274611, 0.241171],
        [0.696115, 0.240167],
        [0.308417, 0.204168],
        [0.666007, 0.203279]])
    assert_array_equal(result.keypoints, expected_keypoints)

def test_get_4_best_keypoint_pairs_not_enough_keypoints_but_fallback_to_previous_frames_possible():
    file_path295 = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_295_not_enough_with_fallback.txt"
    df295 = _read_predictions_and_filter_keypoints(file_path295, is_tracking=True, frame_no=295)
    file_path294 = PATH_PREFIX/"pony_vs_the_killjoys_pool_004_294.txt"
    df294 = _read_predictions_and_filter_keypoints(file_path294, is_tracking=True, frame_no=294)
    df = pd.concat([df294, df295])

    sut = KeypointsExtractor(df, 0.6)
    result = sut.get_4_best_keypoint_pairs(frame_no=295)
    assert result is not None
