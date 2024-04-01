import pytest
import pickle
import os
from src.tasks import tasks
# from unittest.mock import patch

@pytest.mark.parametrize("video_path", [("./tests/data/videos/machine_vs_condors_pool_001-supertiny.mp4")])
# @patch('src.tasks.tasks.video_analysis.update_state')
def test_video_analysis_can_run_prediction_on_video(video_path):
    os.environ["MODEL_DATA_DIR"] = "src/data/model"
    wrapped_results = tasks.video_analysis(video_path)

    assert wrapped_results is not None
    assert isinstance(wrapped_results, dict)
    assert "status" in wrapped_results
    results = wrapped_results["status"]
    assert "1" in results
    assert "2" in results

    assert isinstance(results["1"], list)
    assert isinstance(results["2"], list)
    expected_columns = ["cls", "x", "y", "team", "id"]
    assert set(results["1"][0].keys()).issuperset(set(expected_columns))


@pytest.mark.parametrize("pickled_results_path,expected_total_frames", [
                             ("./tests/data/tracking_set_ultralytics/tiny_tracking_results.pickle", 10),
                             ("./tests/data/tracking_set_ultralytics/machine_vs_condors_pool_001.pickle", 1501)])
def test_video_analysis_final_translation_case_pool_001a(pickled_results_path,expected_total_frames):
    with open(pickled_results_path, "rb") as f:
        tracking_results = pickle.load(f)

    expected_total_frames = expected_total_frames
    df = tasks._translate_coordinates(tracking_results, expected_total_frames)

    assert df is not None
    assert len(df["frame"].unique()) == expected_total_frames
    expected_columns = ["cls", "x", "y", "id", "frame"]
    assert set(df.columns).issuperset(set(expected_columns))
    df_nas = df[df["x"].isna()]
    na_count = len(df_nas)
    print(f"Found {na_count} NAs")
    print(df_nas.head())
    assert na_count == 0
    assert not df.isna().any(axis=None)
    
