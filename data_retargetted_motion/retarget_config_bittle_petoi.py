import numpy as np
from pybullet_utils import transformations

URDF_FILENAME = "bittle/bittle_petoi.urdf"

REF_POS_SCALE = 1
INIT_POS = np.array([0, 0, 0])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=0, axes="sxyz")

POS_SIZE = 3
ROT_SIZE = 4

FRONT_LEFT_HIP_ID = 42
FRONT_LEFT_KNEE_ID = 46
FRONT_LEFT_END_EFFECTOR_ID = 48
BACK_LEFT_HIP_ID = 57
BACK_LEFT_KNEE_ID = 61
BACK_LEFT_END_EFFECTOR_ID = 63
FRONT_RIGHT_HIP_ID = 3
FRONT_RIGHT_KNEE_ID = 7
FRONT_RIGHT_END_EFFECTOR_ID = 9
BACK_RIGHT_HIP_ID = 18
BACK_RIGHT_KNEE_ID = 22
BACK_RIGHT_END_EFFECTOR_ID = 24

# 7, # left hand
# 15, # left foot
# 3, # right hand
# 11 # right foot
SIM_TOE_JOINT_IDS = [
    FRONT_LEFT_END_EFFECTOR_ID,
    BACK_LEFT_END_EFFECTOR_ID,
    FRONT_RIGHT_END_EFFECTOR_ID,
    BACK_RIGHT_END_EFFECTOR_ID
]
# we will do front left, back left, front right, back right
SIM_HIP_JOINT_IDS = [FRONT_LEFT_HIP_ID, BACK_LEFT_HIP_ID, FRONT_RIGHT_HIP_ID, BACK_RIGHT_HIP_ID]

SIM_ROOT_OFFSET = np.array([0, 0, 0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([-0.005, 0.0, 0.0]),
    np.array([-0.005, 0.0, 0.0025]),
    np.array([-0.005, 0.0, 0.0]),
    np.array([-0.005, 0.0, 0.0025])
]

DEFAULT_JOINT_POSE = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0])

JOINT_DAMPING = [0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01]


FORWARD_DIR_OFFSET = np.array([0, 0, 0.0125])
