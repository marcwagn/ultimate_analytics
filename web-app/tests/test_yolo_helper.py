import pytest
import pickle

import ultralytics
from src.tasks import yolo_helper
import numpy as np

@pytest.mark.parametrize("pickled_results_path", [("./tests/data/tracking_set_ultralytics/tiny_tracking_results.pickle")])
def test_yolo_tracking_results_conversion_to_dataframe(pickled_results_path: str):
    tracking_results = _unpickle_tracking_results_and_pad_orig_img(pickled_results_path)
    df = yolo_helper.convert_tracking_results_to_pandas(tracking_results)
    print(df.tail())
    assert df is not None
    assert len(df["frame"].unique()) == 10
    expected_columns = ["cls", "x", "y", "w", "h", "conf", "id", "frame"]
    assert set(df.columns).issuperset(set(expected_columns))

def _unpickle_tracking_results_and_pad_orig_img(pickled_results_path: str) -> list[ultralytics.engine.results.Results]:
    """Unpickles Ultralytics YOLO tracking results and fill orig_img with dummy data"""
    with open(pickled_results_path, "rb") as f:
        tracking_results = pickle.load(f)

    some_img =  np.zeros((2800, 2800, 3))
    assert some_img is not None
    for result in tracking_results:
        if not hasattr(result, "orig_img") or result.orig_img is None:
            setattr(result, "orig_img", some_img)
        assert result.orig_img is not None
    return tracking_results
