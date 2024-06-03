"""
Models a motor as commanded by angular position. Calculates torque based on output of a PD controller.
"""
from typing import Union

from numpy.typing import NDArray
import numpy as np

NUM_MOTORS = 8
MAX_ABS_TORQUE = 0.3
TORQUE_LIMITS = np.asarray([MAX_ABS_TORQUE] * NUM_MOTORS)
DEFAULT_KP = 0.8
DEFAULT_KD = 0.05


class Motor:
    def __init__(self, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD, torque_limits: Union[None, NDArray] = TORQUE_LIMITS):
        self.kp = kp
        self.kd = kd
        self.strength_ratios = [1.0] * NUM_MOTORS
        self.torque_limits = torque_limits

    def convert_motor_command_to_torque(self, motor_command: NDArray, motor_angles: NDArray,
                                        motor_velocities: NDArray):
        additional_torques = np.full(NUM_MOTORS, 0)
        assert len(motor_command) == NUM_MOTORS
        kp = self.kp
        kd = self.kd
        desired_motor_angles = motor_command
        desired_motor_velocities = np.full(NUM_MOTORS, 0)

        motor_torques = -1 * (kp * (motor_angles - desired_motor_angles)) - kd * (
            motor_velocities - desired_motor_velocities) + additional_torques
        motor_torques = self.strength_ratios * motor_torques
        if self.torque_limits is not None:
            if len(self.torque_limits) != len(motor_torques):
                raise ValueError(
                    "Size mismatch between torque limits and motor torques")
            motor_torques = np.clip(motor_torques, -1.0 * self.torque_limits,
                                    self.torque_limits)
        return motor_torques
