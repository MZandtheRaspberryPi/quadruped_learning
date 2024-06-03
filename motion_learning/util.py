from numpy.typing import NDArray
import numpy as np


def quat_norm(quaternion: NDArray):
    q_norm = np.linalg.norm(quaternion)
    if np.isclose(q_norm, 0.0):
        raise ValueError
    return quaternion / q_norm


def standardize_quat(quaternion: NDArray):
    """ q = -q so return quaternion where q.w >= 0"""
    if quaternion[-1] < 0:
        quaternion = -quaternion
    return quaternion
