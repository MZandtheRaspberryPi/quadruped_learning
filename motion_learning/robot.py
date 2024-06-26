
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from numpy.typing import NDArray
import numpy as np
import pybullet

from motor import Motor, MAX_ABS_TORQUE
from sim_config_bittle import (INIT_POS, INIT_ROT, URDF_FILENAME,
                               LOWER_MOTOR_LIMIT, UPPER_MOTOR_LIMIT,
                               SIM_MOTOR_IDS, MOTOR_DIRECTIONS, MOTOR_OFFSETS,
                               DEFAULT_MOTOR_ANGLES)
from util import euler_from_quaternion


class MotorControlMode(Enum):
    POSITION = 0


POS_SIZE = 3
ROT_SIZE = 4


@dataclass
class Observation:
    base_position: NDArray
    base_orientation: NDArray
    base_orientation_euler: NDArray
    base_angular_velocity: NDArray
    motor_angles: NDArray
    motor_velocities: NDArray
    is_safe: bool


@dataclass
class Pose:
    root_position: NDArray  # 3d xyz
    root_orientation: NDArray  # 4d quaternion
    joint_angles: NDArray


DEFAULT_ROBOT_POSE = Pose(INIT_POS, INIT_ROT, DEFAULT_MOTOR_ANGLES)

def check_if_safe(roll, pitch, safe_angle_limit: float = np.pi/4):
    safe = True
    safe &= abs(pitch) < abs(safe_angle_limit)
    safe &= abs(roll) < abs(safe_angle_limit)
    return safe
    


class Robot:
    def __init__(self, pybullet_client, noisy: bool = False, sim_motor_ids: List[int] = SIM_MOTOR_IDS,
                 init_pos=INIT_POS, init_rot=INIT_ROT, lower_motor_limit: float = LOWER_MOTOR_LIMIT,
                 upper_motor_limit: float = UPPER_MOTOR_LIMIT, clip_motor_commands: bool = False,
                 time_step: float = 0.001, motor_directions: Tuple[int] = MOTOR_DIRECTIONS, motor_offsets: Tuple[int] = MOTOR_OFFSETS,
                 urdf_file_name: str = URDF_FILENAME, action_repeat: int = 1, default_pose: Pose = DEFAULT_ROBOT_POSE, render: bool = False,
                 skip_settle_down: bool = False):
        self._pybullet_client = pybullet_client
        self.render = render
        self.skip_settle_down = skip_settle_down
        self.noisy = noisy
        self.quadruped = self._pybullet_client.loadURDF(
            urdf_file_name, init_pos, init_rot, flags=pybullet.URDF_USE_INERTIA_FROM_FILE)
        self.motor_ids = sim_motor_ids
        self.init_pos = init_pos
        self.init_rot = init_rot

        _, self.init_orientation_inv = self._pybullet_client.invertTransform(
            position=[0, 0, 0], orientation=init_rot)
        self.num_motors = len(self.motor_ids)
        self.clip_motor_commands = clip_motor_commands

        self.lower_motor_limit_arr = np.zeros((self.num_motors))
        self.lower_motor_limit_arr[:] = lower_motor_limit
        self.lower_motor_limit = lower_motor_limit

        self.upper_motor_limit_arr = np.zeros((self.num_motors))
        self.upper_motor_limit_arr[:] = upper_motor_limit
        self.upper_motor_limit = upper_motor_limit

        self.state_action_counter = 0
        self.time_step = time_step

        self.motor_directions = motor_directions
        self.motor_offsets = motor_offsets

        self.motor_controller = Motor()
        self.last_state_time = None
        self.last_action_time = None

        self.default_pose = default_pose

        self.action_repeat = action_repeat
        self.joint_name_to_id = {}

    def settle_down_for_reset(self, reset_time: float = 0.5, default_motor_angles: NDArray = DEFAULT_ROBOT_POSE):

        if reset_time <= 0:
            return

        num_steps_to_reset = int(reset_time / self.time_step)
        for _ in range(num_steps_to_reset):
            observation = self.step(default_motor_angles)
            # here env won't render so we do
            if self.render:
                self._pybullet_client.configureDebugVisualizer(
                    pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,
                    1)

    def set_pose(self, pose: Pose):
        num_joints = len(pose.joint_angles)
        pybullet.resetBasePositionAndOrientation(
            self.quadruped, pose.root_position, pose.root_orientation)

        for j in range(num_joints):
            joint_index = self.motor_ids[j]
            joint_target_angle = pose.joint_angles[j]
            j_info = pybullet.getJointInfo(self.quadruped, joint_index)
            j_state = pybullet.getJointStateMultiDof(
                self.quadruped, joint_index)
            j_pose_size = len(j_state[0])
            j_vel_size = len(j_state[1])

            if (j_pose_size > 0):
                j_pose = np.array([joint_target_angle])
                j_vel = np.zeros(j_vel_size)
                pybullet.resetJointStateMultiDof(
                    self.quadruped, joint_index, j_pose, j_vel)

    def reset(self, reset_time: float = 0.2):

        self._pybullet_client.resetBasePositionAndOrientation(
            self.quadruped, self.init_pos,
            self.init_rot)
        self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                                [0, 0, 0])

        # remove joint dampening
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(joint_info[0],
                                                 -1,
                                                 linearDamping=0,
                                                 angularDamping=0)

        self.set_pose(self.default_pose)
        if self.render:
            self._pybullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,
                1)
        self.build_joint_name_dict()
        self.state_action_counter = 0
        if not self.skip_settle_down:
            self.settle_down_for_reset(reset_time, self.default_pose.joint_angles)
        # this seems important, perhaps there's a default motor controller in pybullet
        for joint_index in self.motor_ids:
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_index,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        # self._pybullet_client.setJointMotorControlArray(self.quadruped,
        #                                                 self.motor_ids,
        #                                                 pybullet.VELOCITY_CONTROL,
        #                                                 forces=np.zeros((len(self.motor_ids))))

    def transform_angular_velocity_to_local_frame(self, angular_velocity,
                                            orientation):
        """Transform the angular velocity from world frame to robot's frame.

        Args:
            angular_velocity: Angular velocity of the robot in world frame.
            orientation: Orientation of the robot represented as a quaternion.

        Returns:
            angular velocity based on the given orientation.
        """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform(
            [0, 0, 0], orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def get_observation(self) -> Observation:
        """returns the observations including orientation (4d quaternion), and joint angles in the order specified by joint_ids
        """
        if self.noisy:
            raise NotImplementedError
        joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self.motor_ids)

        base_position, orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        # Computes the relative orientation relative to the robot's
        # initial_orientation.
        _, base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=orientation,
            positionB=[0, 0, 0],
            orientationB=self.init_orientation_inv)
        base_angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
        base_angular_velocity = self.transform_angular_velocity_to_local_frame(base_angular_velocity,
                                                 base_orientation)

        motor_angles = [state[0] for state in joint_states]
        motor_angles = np.multiply(np.asarray(
            motor_angles) - np.asarray(self.motor_offsets), np.asarray(self.motor_directions))

        motor_velocities = [state[1] for state in joint_states]
        motor_velocities = np.multiply(
            np.asarray(motor_velocities), np.asarray(self.motor_directions))
        self.last_state_time = self.state_action_counter * self.time_step

        # observation.extend(self.GetTrueMotorTorques())
        # self._motor_model.convert_to_torque(
        # motor_commands, q, qdot, qdot_true, control_mode)
        # observation.extend(self.GetTrueBaseOrientation())
        # observation.extend(self.GetTrueBaseRollPitchYawRate())
        roll, pitch, yaw = euler_from_quaternion(base_orientation[0], base_orientation[1],
                                                 base_orientation[2], base_orientation[3])
        is_safe = check_if_safe(roll, pitch, np.pi/4)
        obs = Observation(base_position, base_orientation, np.array([roll, pitch, yaw]), base_angular_velocity,
                          motor_angles, motor_velocities, is_safe)
        return obs

    def apply_action(self, motor_commands: NDArray):
        # , observation: Observation

        if self.clip_motor_commands:
            motor_commands = np.where(
                motor_commands > self.upper_motor_limit, self.upper_motor_limit_arr, motor_commands)
            motor_commands = np.where(
                motor_commands < self.lower_motor_limit, self.lower_motor_limit_arr, motor_commands)

        # self.last_action_time = self.state_action_counter * self.time_step

        # motor_angles = observation.motor_angles

        # motor_velocities = observation.motor_velocities

        # torque_to_command = self.motor_controller.convert_motor_command_to_torque(motor_commands, motor_angles,
        #                                                                           motor_velocities)
        # applied_motor_torque = np.multiply(
        #     torque_to_command, self.motor_directions)
        # self._pybullet_client.setJointMotorControlArray(
        #     bodyIndex=self.quadruped,
        #     jointIndices=self.motor_ids,
        #     controlMode=self._pybullet_client.TORQUE_CONTROL,
        #     forces=applied_motor_torque)
        self._pybullet_client.setJointMotorControlArray(
                self.quadruped,
                SIM_MOTOR_IDS,
                self._pybullet_client.POSITION_CONTROL,
                motor_commands,
                forces=[MAX_ABS_TORQUE]*len(motor_commands),
        )

    def step(self, motor_commands: "np.array"):
        for i in range(self.action_repeat):
            self.apply_action(motor_commands=motor_commands) # , observation=new_observation
            self._pybullet_client.stepSimulation()
            new_observation = self.get_observation()
        self.state_action_counter += 1
        return new_observation

    def build_joint_name_dict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self.joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self.joint_name_to_id[joint_info[1].decode(
                "UTF-8")] = joint_info[0]
