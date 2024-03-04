from os import path
from src.ultimate_pipeline.pipelines.data_processing.supervisely_converter import convert_to_normalized_bounding_boxes
import numpy as np
from numpy.testing import assert_array_equal

def test_convert_video_annotations_with_object_keys():
    prefix = './src/tests/data/supervisely/sample_video_1'
    annotations_filename = path.join(prefix, "ds0/ann/machine_vs_condors_pool_001a.mp4.json")

    df = convert_to_normalized_bounding_boxes(annotations_filename)

    width = 3840.0
    height = 2160.0

    assert len(df) == 2211
    assert df.iloc[0].at["cls"] == 0
    assert df.iloc[0].at["x"] == 0.38033854166666664
    assert df.iloc[0].at["y"] == 0.3958333333333333
    assert df.iloc[0].at["w"] == 0.02109375
    assert df.iloc[0].at["h"] == 0.08518518518518518

    actualFirstRow = df.iloc[0, 0:5].to_numpy()
    expectedFirstRow = np.array([0, (1501+1420)/2/width, (947+763)/2/height, (1501-1420)/width, (947-763)/height])
    assert_array_equal(actualFirstRow, expectedFirstRow)

    actualFourteenthRow = df.iloc[13, 0:5].to_numpy()
    expectedFourteenthRow = np.array([0, (2084+2049)/2/width, (438+343)/2/height, (2084-2049)/width, (438-343)/height])
    assert_array_equal(actualFourteenthRow, expectedFourteenthRow)

    known_disc_object_key = "6e87e4b4096144279ba6b9306ad10aac"
    actualFirstRowWithDisc = df[df["object_key"]==known_disc_object_key]
    actualFirstRowWithDisc = actualFirstRowWithDisc.iloc[0,0:5].to_numpy()
    expectedFirstRowWithDisc = np.array([1, (2160+2140)/2/width, (450+433)/2/height, (2160-2140)/width, (450-433)/height])
    assert_array_equal(actualFirstRowWithDisc, expectedFirstRowWithDisc)


def test_convert_video_annotations_with_object_ids():
    prefix = './src/tests/data/supervisely/sample_video_2'
    annotations_filename = path.join(prefix, "annotations/sockeye_vs_rhino_slam_pool_004.json")

    df = convert_to_normalized_bounding_boxes(annotations_filename)

    width = 3840.0
    height = 2160.0

    assert len(df) == 2543
    actualFirstRow = df.iloc[0, 0:5].to_numpy()
    expectedFirstRow = np.array([0, (464+327)/2/width, (1283+1106)/2/height, (464-327)/width, (1283-1106)/height])
    assert_array_equal(actualFirstRow, expectedFirstRow)
