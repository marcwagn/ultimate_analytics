import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from src.tasks.keypoints import KeypointQuad
import src.tasks.homography as hg


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

def generate_sample_keypoints_and_real_pitch_expected_values():
    return [
            ([0.152733, 0.375162], [0, 18]),
            ([0.825292, 0.372376], [37, 18]),
            ([0.230659, 0.288993], [0, 0]),
            ([0.753578, 0.285994], [37, 0])
        ]

@pytest.mark.parametrize("h", generate_sample_homography_matrix())
@pytest.mark.parametrize("keypoints,expected", generate_sample_keypoints_and_real_pitch_expected_values())
def test_convert_2d_point(h, keypoints, expected):
    result = hg.convert_h(h, np.float32(keypoints))
    assert_array_almost_equal(result, np.float32(expected), decimal=6)
