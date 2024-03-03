import pytest
from os import path
from src.features.supervisely import VideoAnnotationsConverter
import json
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

def test_convert_to_yolo():
    prefix = './test/data/supervisely/sample_video_1'
    annotations_filename = path.join(prefix, "ds0/ann/machine_vs_condors_pool_001a.mp4.json")

    sup = VideoAnnotationsConverter(annotations_filename)
    df = sup.read_bounding_boxes_dataframe()

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
