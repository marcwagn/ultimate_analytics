import cv2
import numpy as np
from .keypoints import KeypointQuad

keypoints_to_known_coordinates_map = {
    31: dict(real_pitch=np.float32([0, 0])),
    32: dict(real_pitch=np.float32([37, 0])),
    34: dict(real_pitch=np.float32([0, 18])),
    35: dict(real_pitch=np.float32([37, 18])),
    39: dict(real_pitch=np.float32([0, 82])),
    40: dict(real_pitch=np.float32([37, 82])),
    41: dict(real_pitch=np.float32([0, 100])),
    42: dict(real_pitch=np.float32([37, 100]))
  }

def calculate_homography_matrix(keypoints:KeypointQuad) -> np.ndarray:
    """
    Calculate the homography matrix.
    Args:
        keypoints (KeypointQuad): A series of 4 pitch keypoints extracted from a specific image/video frame
    Return:
        np.ndarray (3,3)
    """
    src_points = keypoints.keypoints
    dest_map_key = "real_pitch"
    dst_points = np.float32([keypoints_to_known_coordinates_map[cls_id][dest_map_key] for cls_id in keypoints.cls_ids])
    H, _ = cv2.findHomography(src_points, dst_points)
    return H


def convert_h(H, vec):
    """
    Converts point coordinates (x, y) to new coordinates using homography matrix.
    Args:
        H (np.ndarray): A homography matrix (3,3)
        vec (np.ndarray): A vector of size (2,) representing a single point to convert
    Return:
        np.ndarray of shape (2,)
    """
    m = vec.shape[0]
    if m != 2 or vec.ndim > 1:
        raise ValueError("Input vector has to be of size (2,)")
    # Add z coordinate to make the vector/matrix compatible with the 3d homography matrix
    vec = np.r_[vec, [1]]
    
    pt = np.dot(H, vec)
    # Normalize the output vector/matrix (we want 1 in the last row, i.e. z=1)
    pt_norm = pt/pt[2]
    return pt_norm[0:2]
