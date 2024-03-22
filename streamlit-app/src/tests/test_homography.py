import numpy as np
from numpy.testing import assert_array_almost_equal
from src.keypoints import KeypointQuad
import src.homography as hg

def test_calculate_homography_matrix():
    keypoints_mid_points = np.float32([
        [0.152733, 0.375162],
        [0.825292, 0.372376],
        [0.230659, 0.288993],
        [0.753578, 0.285994]])
    quad = KeypointQuad(
        keypoint_line_keys=["TF", "TC"], 
        cls_ids=[34,35,31,32],
        keypoints=keypoints_mid_points)
    
    h = hg.calculate_homography_matrix(keypoints=quad)
    assert h is not None
    assert h.shape == (3,3)

    result = hg.convert_h(h, np.array([0.152733, 0.375162, 1]))
    assert result is not None

    assert_array_almost_equal(result, np.float32([18,0,1]), decimal=6)
