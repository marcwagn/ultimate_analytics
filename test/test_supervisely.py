import pytest
from os import path
import src.features.supervisely as sup
import json
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

def test_convert_to_yolo():
    prefix = './test/data/supervisely/sample1'
    key_id_filename = path.join(prefix, "key_id_map.json")
    with open(key_id_filename, 'r') as f:
        key_id_map =  json.load(f)
    assert key_id_map is not None
    annotations_filename = path.join(prefix, "ds0/ann/machine_vs_condors_pool_001a.mp4.json")

    df = sup.convert_to_yolo_df(key_id_filename, annotations_filename)

    assert len(df) == 2211
    assert df.iloc[0].at["cls"] == 0
    assert df.iloc[0].at["x"] == 0.38033854166666664
    assert df.iloc[0].at["y"] == 0.3958333333333333
    assert df.iloc[0].at["w"] == 0.02109375
    assert df.iloc[0].at["h"] == 0.08518518518518518

    actualfirstRow = df.iloc[0, 0:5].to_numpy()
    expectedFirstRow = np.array([0, 0.38033854166666664, 0.3958333333333333, 0.02109375, 0.08518518518518518])
    assert_array_equal(actualfirstRow, expectedFirstRow)
    #0,0.44388020833333336,0.17962962962962964,0.012239583333333333,0.041666666666666664,9,0
