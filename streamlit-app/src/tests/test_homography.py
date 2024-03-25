import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from src.keypoints import KeypointQuad
import src.homography as hg


def generate_sample_keypoints():
    keypoints_mid_points = np.float32(
        [
            [0.152733, 0.375162],
            [0.825292, 0.372376],
            [0.230659, 0.288993],
            [0.753578, 0.285994],
        ]
    )
    yield KeypointQuad(
        keypoint_line_keys=["TF", "TC"],
        cls_ids=[34, 35, 31, 32],
        keypoints=keypoints_mid_points,
    )

@pytest.mark.parametrize("keypoints", generate_sample_keypoints())
def test_calculate_homography_matrix(keypoints):
    h = hg.calculate_homography_matrix(keypoints=keypoints)
    assert h.shape == (3, 3)

def generate_sample_homography_matrix():
    keypoints = list(generate_sample_keypoints())[0]
    h = hg.calculate_homography_matrix(keypoints)
    yield h

@pytest.mark.parametrize("h", generate_sample_homography_matrix())
def test_convert_2d_point(h):
    result = hg.convert_h(h, np.array([0.152733, 0.375162]))
    assert_array_almost_equal(result, np.float32([18, 0]), decimal=6)

@pytest.mark.parametrize("h", generate_sample_homography_matrix())
def test_convert_2d_vector(h):
    sample_points = np.float32(
        [
            [0.152733, 0.375162],
            [0.825292, 0.372376],
            [0.230659, 0.288993],
            [0.753578, 0.285994],
        ]
    ).T
    expected_points = np.float32(
       [
        [18, 37],
        [100, 37],
        [18, 0],
        [100,0]
       ]
    ).T
    result = hg.convert_h(h, sample_points)
    assert result.shape == expected_points.shape
    assert_array_almost_equal(result, expected_points, decimal=6)