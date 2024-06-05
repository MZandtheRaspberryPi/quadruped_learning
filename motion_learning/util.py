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

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians