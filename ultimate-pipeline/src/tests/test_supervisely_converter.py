import json
from os import path

import pytest
import src.ultimate_pipeline.pipelines.data_processing.supervisely_converter as sc
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

def test_convert_video_annotations_with_object_keys():
    prefix = "./src/tests/data/supervisely/sample_video_1"
    annotations_filename = path.join(prefix, "ds0/ann/machine_vs_condors_pool_001a.mp4.json")

    df = sc.convert_video_annotations(annotations_filename)

    width = 3840.0
    height = 2160.0

    assert df is not None
    assert isinstance(df, pd.DataFrame)
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
    prefix = "./src/tests/data/supervisely/sample_video_2"
    annotations_filename = path.join(prefix, "annotations/sockeye_vs_rhino_slam_pool_004.json")

    df = sc.convert_video_annotations(annotations_filename)

    width = 3840.0
    height = 2160.0

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2543
    actualFirstRow = df.iloc[0, 0:5].to_numpy()
    expectedFirstRow = np.array([0, (464+327)/2/width, (1283+1106)/2/height, (464-327)/width, (1283-1106)/height])
    assert_array_equal(actualFirstRow, expectedFirstRow)

def test_convert_single_image_annotation():
    prefix = "./src/tests/data/supervisely/sample_images_1"
    annotations_file = path.join(prefix, "test/ann", "sockeye_vs_rhino_slam_pool_004.mp4_00000.jpg.json")
    metadata_file = path.join(prefix, "meta.json")

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        meta_map = { m["id"]: i for i, m in enumerate(metadata["classes"])}
    
    df = sc.convert_single_image_annotation_to_detect_data(annotations, frame_key=0, meta_map=meta_map)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    width = 3840.0
    height = 2160.0

    actualFirstRow = df.iloc[0, 0:5].to_numpy()
    expectedClassIdPerson = 13
    expectedFirstRow = np.array([expectedClassIdPerson, (464+327)/2/width, (1283+1106)/2/height, (464-327)/width, (1283-1106)/height])
    assert_array_equal(actualFirstRow, expectedFirstRow)

    actualLastRow = df.iloc[-1, 0:5].to_numpy()
    expectedClassIdBrf = 6
    expectedLastRow = np.array([expectedClassIdBrf, (3796+3755)/2/width, (1185+1144)/2/height, (3796-3755)/width, (1185-1144)/height])
    assert_array_equal(actualLastRow, expectedLastRow)

def test_convert_image_annotation_folder_nonexistent():
    prefix = "./src/tests/data/supervisely/sample_images_nonexistent"
    annotations_folder = path.join(prefix, "test/ann")
    metadata_file = path.join(prefix, "meta.json")

    with pytest.raises(ValueError, match="If source is passed as string, it needs to represent an existing directory"):
        sc.convert_images_annotations_folder_to_detect_data(annotations_folder, meta_file=metadata_file)

def test_convert_image_annotation_folder():
    prefix = "./src/tests/data/supervisely/sample_images_1"
    annotations_folder = path.join(prefix, "test/ann")
    metadata_file = path.join(prefix, "meta.json")

    df = sc.convert_images_annotations_folder_to_detect_data(annotations_folder, meta_file=metadata_file)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 780
    assert len(df.groupby("frame")) == 30

def test_convert_single_image_annotation_file_to_pose_estimation():
    prefix = "./src/tests/data/supervisely/sample_images_2"
    annotations_file = path.join(prefix, "mini_test_set/ann", "machine_vs_condors_pool_006_0.jpg.json")
    metadata_file = path.join(prefix, "meta.json")

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    df = sc.convert_single_image_annotation_file_to_pose_estimation(annotations, "machine_vs_condors_pool_006_0.jpg", meta_map=metadata)

    # Derived from the respective annotation json
    width = 3840.0
    height = 2160.0
    expected_unscaled_triplets = [(1260, 432, 2), (2568, 424, 2), (1913, 462, 2), (1148, 502, 2), (2682, 494, 2), (1921, 594,2), (1923, 692, 2), (1929, 841, 2), (55, 1193, 2), (3829, 1173, 2), (0, 0, 0), (0, 0, 0), (1952, 1494, 2)]
    leftX = 55.0
    rightX = 3829.0
    topY = 424.0
    bottomY = 1494.0
    expected_object_width = (rightX-leftX)/width
    expected_object_height = (bottomY-topY)/height
    expected_centre_x = (rightX+leftX)/2/width
    expected_centre_y = (bottomY+topY)/2/height

    def generateExpectedKeypointsSequence():
        for (x, y, vis) in expected_unscaled_triplets:
            scaled_x, scaled_y = x/width, y/height
            yield scaled_x
            yield scaled_y
            yield vis

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    actual_row = df.iloc[0,:]

    assert actual_row["cls"] == 0
    assert actual_row["x"] == expected_centre_x
    assert actual_row["y"] == expected_centre_y
    assert actual_row["w"] == expected_object_width
    assert actual_row["h"] == expected_object_height

    actual_keypoint_coords = actual_row[5:-1].to_numpy()
    expected_keypoint_coords = np.array(list(generateExpectedKeypointsSequence()))
    assert len(actual_keypoint_coords) ==  len(expected_keypoint_coords)
    assert_array_equal(actual_keypoint_coords, expected_keypoint_coords)

    assert actual_row["frame"] == "machine_vs_condors_pool_006_0.jpg"


def test_convert_image_annotation_folder_to_pose_estimation():
    prefix = "./src/tests/data/supervisely/sample_images_2"
    annotations_folder = path.join(prefix, "mini_test_set/ann")
    metadata_file = path.join(prefix, "meta.json")

    df = sc.convert_images_annotations_folder_to_pose_data(annotations_folder, meta_file=metadata_file)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert len(df.groupby("frame")) == 5

    fixed_column_list = ["cls", "x", "y", "w", "h", "frame"]
    for c in fixed_column_list:
        del df[c]

    assert len(df.columns) == 13*3