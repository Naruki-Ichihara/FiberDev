import cv2 as cv
import numpy as np
import cupy as cp

def crop_2D(start: tuple[int, int], end: tuple[int, int], image: cp.ndarray) -> cp.ndarray:
    """Crop image.

    Args:
        start (tuple[int, int]): Start coordinate.
        end (tuple[int, int]): End coordinate.
        image (np.ndarray): Image.

    Returns:
        np.ndarray: Cropped image.

    """
    return image[start[0]:end[0], start[1]:end[1]]

def drop_edges_3D(width: int, volume: cp.ndarray) -> cp.ndarray:
    """Drop edges of 3D volume.

    Args:
        width (int): Width of edges.
        volume (np.ndarray): 3D volume.

    Returns:
        np.ndarray: 3D volume without edges.

    """
    return volume[width:volume.shape[0]-width, width:volume.shape[1]-width, width:volume.shape[2]-width]