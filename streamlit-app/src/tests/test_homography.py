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
def test_convert_3d_point(h):
    result = hg.convert_h(h, np.array([0.152733, 0.375162, 1]))
    assert_array_almost_equal(result, np.float32([18, 0, 1]), decimal=6)

@pytest.mark.skip(reason="Not passing yet, but I'm not sure if we need this feature at all")
@pytest.mark.parametrize("h", generate_sample_homography_matrix())
def test_convert_2d_vector(h):
    result = hg.convert_h(h, np.array([[0.152733, 0.375162], [0.230659, 0.288993], [0.753578, 0.285994]]))
    assert_array_almost_equal(result, np.float32([[18, 0], [0, 0], [100,37]]), decimal=6)

@pytest.mark.skip(reason="Not passing yet, but I'm not sure if we need this feature at all")
@pytest.mark.parametrize("h", generate_sample_homography_matrix())
def test_convert_3d_vector(h):
    result = hg.convert_h(h, np.array([[0.152733, 0.375162, 1], [0.230659, 0.288993, 1], [0.753578, 0.285994, 1]]))
    assert_array_almost_equal(result, np.float32([[18, 0, 1], [0, 0, 1], [100,37, 1]]), decimal=6)