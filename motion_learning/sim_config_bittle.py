import os

import numpy as np
from pybullet_utils import transformations

URDF_FILENAME = os.path.join(os.path.dirname(
    __file__), "..", "bittle", "bittle_petoi.urdf")

REF_POS_SCALE = 1
# INIT_POS = np.array([0, 0, 0.15])
INIT_POS = np.array([0, 0, 0.2])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=0, axes="sxyz")

DEFAULT_MOTOR_ANGLES = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
"""
PETOI_URDF

[(3, 'shrfs_joint'), (7, 'shrft_joint'), (18, 'shrrs_joint'), (22, 'shrrt_joint'), (33, 'neck_joint'), (42, 'shlfs_joint'), (46, 'shlft_joint'), (57, 'shlrs_joint'), (61, 'shlrt_joint')]

end_effector_rf - 9
end_effector_rr - 24
end_effector_lf - (48, 'Rigid 91')]
end_effector_lr - [(63, 'Rigid 92')]
"""
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

SIM_MOTOR_IDS = [FRONT_LEFT_HIP_ID, FRONT_LEFT_KNEE_ID,
                 BACK_LEFT_HIP_ID, BACK_LEFT_KNEE_ID, FRONT_RIGHT_HIP_ID, FRONT_RIGHT_KNEE_ID,
                 BACK_RIGHT_HIP_ID, BACK_RIGHT_KNEE_ID]
LOWER_MOTOR_LIMIT = -np.pi/2
UPPER_MOTOR_LIMIT = np.pi/2

MOTOR_DIRECTIONS = (1, 1, 1, 1, 1, 1, 1, 1)
MOTOR_OFFSETS = (0, 0, 0, 0, 0, 0, 0, 0)
