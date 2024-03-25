import cv2
import numpy as np
from src.keypoints import KeypointQuad

keypoints_to_known_coordinates_map = {
    31: dict(real_pitch=np.float32([0,0])),
    32: dict(real_pitch=np.float32([0,37])),
    34: dict(real_pitch=np.float32([18,0])),
    35: dict(real_pitch=np.float32([18,37])),
    39: dict(real_pitch=np.float32([82,0])),
    40: dict(real_pitch=np.float32([82,37])),
    41: dict(real_pitch=np.float32([100,0])),
    42: dict(real_pitch=np.float32([100,37]))
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
        vec (np.ndarray): A vector of size (2,) a matrix of size (2,n), representing a single or multiple points to convert
    Return:
        np.ndarray of shape that matches vec
    """
    m = vec.shape[0]
    if m != 2 or vec.ndim > 2:
        raise ValueError("Input vector has to be of size (2,) or (2,n).")
    n = vec.shape[1] if vec.ndim == 2 else 1
    # Add z coordinate to make the vector/matrix compatible with the 3d homography matrix
    ones = [1] if vec.ndim==1 else np.ones(n).reshape(1,n)
    vec = np.r_[vec, ones]
    
    pt = np.dot(H, vec)
    # Normalize the output vector/matrix (we want 1 in the last row, i.e. z=1)
    if vec.ndim == 1:
        pt_norm = pt/pt[2]
        return pt_norm[0:2]
    else:
        t = np.eye(3)

        t[-1,:] = np.array([1/sum(pt[-1,:])]*3)
        pt_norm = np.dot(t, pt)
        return pt_norm