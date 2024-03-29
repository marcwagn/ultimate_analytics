import pytest
import pandas as pd
from src.tasks.keypoints import KeypointsExtractor
from src.tasks.homography import calculate_homography_matrix
from io import BytesIO
import zipfile


def _read_prediction_dataframe_from_txt_archive(archive_path: str, is_tracking=True) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path, 'r') as archive:
        file_list = sorted(archive.namelist(), key=lambda x: int(x.rstrip(".txt").split("_")[-1]))
        dfs = []
        for i, file_name in enumerate(file_list):
            columns=["cls", "x", "y", "w", "h", "conf", "id"] if is_tracking else ["cls", "x", "y", "w", "h", "conf"]
            with BytesIO(archive.read(file_name)) as file_buffer:
                df = pd.read_csv(file_buffer, sep=" ", header=None, names=columns)
            df["frame"] = i
            df["filename"] = file_name
            dfs.append(df)

        return pd.concat(dfs)


@pytest.mark.parametrize("labels_archive_path", [("./tests/data/tracking_set_large_zip/labels.zip")])
def test_get_4_best_keypoint_pairs_on_real_folder(labels_archive_path):
    df = _read_prediction_dataframe_from_txt_archive(labels_archive_path)
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