import numpy as np
from pybullet_utils import transformations

URDF_FILENAME = "bittle/bittle_simple.urdf"

REF_POS_SCALE = 1
INIT_POS = np.array([0, 0, 0])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=0, axes="sxyz")

POS_SIZE = 3
ROT_SIZE = 4

# 7, # left hand
# 15, # left foot
# 3, # right hand
# 11 # right foot
SIM_TOE_JOINT_IDS = [
    8,
    11,
    2,
    5
]
# we will do front left, back left, front right, back right
SIM_HIP_JOINT_IDS = [6, 9, 0, 3]

SIM_ROOT_OFFSET = np.array([0, 0, 0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([-0.005, 0.0, 0.0]),
    np.array([-0.005, 0.0, 0.0025]),
    np.array([-0.005, 0.0, 0.0]),
    np.array([0.005, 0.0, 0.0025])
]

DEFAULT_JOINT_POSE = np.array(
    [0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145, 0, 0, 0, 0])

JOINT_DAMPING = [0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01]


FORWARD_DIR_OFFSET = np.array([0, 0, 0.0125])
