import pytest
import pickle
from src.tasks import yolo_helper

@pytest.mark.parametrize("pickled_results_path", [("./tests/data/tracking_set_ultralytics/tracking_results.pickle")])
def test_yolo_tracking_results_conversion_to_dataframe(pickled_results_path: str):
    with open(pickled_results_path, "rb") as f:
        tracking_results = pickle.load(f)

    assert tracking_results is not None
    df = yolo_helper.convert_tracking_results_to_pandas(tracking_results)
    print(df.tail())
    assert df is not None
    assert len(df["frame"].unique()) == 10
    expected_columns = ["cls", "x", "y", "w", "h", "conf", "id", "frame"]
    assert set(df.columns).issuperset(set(expected_columns))
