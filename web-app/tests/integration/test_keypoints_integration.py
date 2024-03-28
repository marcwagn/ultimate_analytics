import pytest
import pandas as pd
from src.tasks.keypoints import KeypointsExtractor
from src.tasks.homography import calculate_homography_matrix
import os

def _read_prediction_dataframe_from_dir(folder: str, is_tracking=True) -> pd.DataFrame:
    file_list = sorted(os.listdir(folder), key=lambda x: int(x.rstrip(".txt").split("_")[-1]))
    dfs = []
    for i, file_name in enumerate(file_list):
        columns=["cls", "x", "y", "w", "h", "conf", "id"] if is_tracking else ["cls", "x", "y", "w", "h", "conf"]
        df = pd.read_csv(os.path.join(folder, file_name), sep=" ", header=None, names=columns)
        df["frame"] = i
        df["filename"] = file_name
        dfs.append(df)

    return pd.concat(dfs)


# In order to run the test, make sure that it points to a folder with YOLO8 object tracking results (txt files) and remove the skip decorator.
@pytest.mark.skip("Integration tests, not part of unit tests")
@pytest.mark.parametrize("folder_path", [("../ultimate-pipeline/runs/detect/track6/labels")])
def test_get_4_best_keypoint_pairs_on_real_folder(folder_path):
    df = _read_prediction_dataframe_from_dir(folder_path)
    expected_detections = 64057
    expected_video_frames = 2935

    assert len(df) == expected_detections

    unprocessable_frames = []
    sut = KeypointsExtractor(df, conf_threshold=0.5, max_lookback=30)
    for i in range(1, expected_video_frames):
        try:
            result = sut.get_4_best_keypoint_pairs(frame_no=i)
        except ValueError:
            unprocessable_frames.append(i)

        assert result is not None, f"result is None for frame {i}"

        H = calculate_homography_matrix(result)
        assert H is not None, f"Homography matrix is None for frame {i}"
        assert H.shape == (3,3)
    assert len(unprocessable_frames) == 0