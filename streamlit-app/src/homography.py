import cv2
import numpy as np
#from enum import Enum
from src.keypoints import KeypointQuad

#class CoordinatesType

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
    Converts a vector to new coordinates using homography matrix.
    Args:
        H (np.ndarray): A homography matrix (3,3)
        vec (np.ndarray): A vector of size (,2) or (,3), or a matrix of size (n,2) or (n,3)
    """
    # m, n = vec.shape
    # if n == 2:
    #     # Add z coordinate to make the matrix 3d
    #     vec = np.c_[vec, np.ones(m)]
    pt = np.dot(H, vec)
    return pt/pt[2]