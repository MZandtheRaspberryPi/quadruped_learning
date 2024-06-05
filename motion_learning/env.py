import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tempfile

from dataclasses import dataclass
import time
from typing import List, Callable, Union, Tuple
from tqdm.auto import tqdm

from numpy.typing import NDArray
import numpy as np
from numpy.typing import ArrayLike
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from robot import Robot, MotorControlMode, Observation, URDF_FILENAME, INIT_POS, INIT_ROT, Pose, SIM_MOTOR_IDS, DEFAULT_MOTOR_ANGLES
from ref_motion_utils import load_ref_motions, POS_SIZE, ROT_SIZE
from util import euler_from_quaternion

NUM_SIMULATION_ITERATION_STEPS = 300
GROUND_URDF_FILENAME = "plane_implicit.urdf"

# our observed state will use roll, pitch, roll rate, pitch rate, 8 joint angles we are targeting next time step
# and calculate rewards using this
OBSERVED_STATE_SIZE = 2 + 2 + 8


@dataclass
class SimParams:
    """params for the pybullet sim"""
    sim_time_step: float = 0.005
    num_action_repeat: int = 33
    enable_hard_reset: bool = False
    enable_rendering: bool = False
    enable_rendering_gui: bool = True
    robot_on_track: bool = False
    camera_distance: float = 0.5
    camera_yaw: float = 0
    camera_pitch: float = -30
    render_width: int = 480
    render_height: int = 360
    egl_rendering: bool = False
    # 1 is position mode for joints, controling by specifying angle directly
    motor_control_mode: int = MotorControlMode.POSITION
    reset_time: float = -1
    enable_action_filter: bool = True
    enable_action_interpolation: bool = True
    allow_knee_contact: bool = False
    enable_clip_motor_commands: bool = True

def observed_state_to_vector(observation: Observation):
   orientation = observation.base_orientation_euler
   base_angular_vels = observation.base_angular_velocity
   joint_angles = observation.motor_angles
   return np.hstack([orientation[0:2], base_angular_vels[0:2], joint_angles]).astype(np.float32)


def expert_trajectory_to_states_actions(expert_trajectory: ArrayLike, num_repeats: int = 5) -> Tuple[int, ArrayLike, ArrayLike]:
    """returns the number of frames before a loop, as well as an array like entry of stuff to calculate
    rolled out trajectories with.

    Args:
        expert_trajectory (ArrayLike): _description_
        num_repeats (int, optional): _description_. Defaults to 5.

    Returns:
        Tuple[Dict[float, ArrayLike], ArrayLike]: _description_
    """
    num_frames = expert_trajectory.shape[0]
    num_new_frames = num_frames * num_repeats
    phases = np.linspace(0, 1, num_frames)
    phases = np.hstack([phases] * num_repeats)
    phases = phases.reshape((phases.shape[0], 1))
    all_frames = np.vstack([expert_trajectory] * num_repeats)
    all_frames = np.hstack([phases, all_frames])
    # get the angles, which starts after phase, pos, rotation
    num_angles = all_frames[0, 1 + POS_SIZE + ROT_SIZE:].shape[0]
    all_actions = np.vstack([all_frames[1:, 1 + POS_SIZE + ROT_SIZE:], all_frames[0, 1 + POS_SIZE + ROT_SIZE:].reshape((1, num_angles))])
    return num_frames, all_frames, all_actions


def all_states_to_expert_state_vector(all_frames: ArrayLike, timestep: float):
    expert_observed_states = np.zeros((all_frames.shape[0], OBSERVED_STATE_SIZE))
    for i in range(all_frames.shape[0]):
        if i == 0:
            prev_index = all_frames.shape[0] - 1
        else:
            prev_index = i - 1
        prev_orientation = all_frames[prev_index, 1 + POS_SIZE:1+POS_SIZE + ROT_SIZE]
        orientation = all_frames[i, 1 + POS_SIZE:1+POS_SIZE + ROT_SIZE]
        prev_roll, prev_pitch, prev_yaw = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        roll, pitch, yaw = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        roll_rate = roll / timestep
        pitch_rate = pitch / timestep
        cur_motor_angles = all_frames[i, 1 + POS_SIZE + ROT_SIZE:]
        expert_observed_states[i, 0:2] = np.array([roll, pitch])
        expert_observed_states[i, 2:4] = np.array([roll_rate, pitch_rate])
        expert_observed_states[i, 4:] = cur_motor_angles
    return expert_observed_states

class SimEnv:
    def __init__(self, config: SimParams, reference_motions: List[NDArray], show_reference_model_flag: bool = False,
                 discount_factor: float = 0.97, num_motion_repeats: int = 3):
        """_summary_

        Args:
            config (SimParams): _description_
            task (Callable): callable to calc reward and termination condition, takes sim env as arg
        """
        self.config = config
        self.render_flag: bool = self.config.enable_rendering
        self.discount_factor = discount_factor

        self.num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS /
                                                self.config.num_action_repeat)
        self.env_timestep = self.config.num_action_repeat * self.config.sim_time_step

        self.ground = None
        self._robot: Robot = None
        self.env_step_counter = 0

        if self.render_flag:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
            self._pybullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_RENDERING,
                self.config.enable_rendering_gui)
            self._pybullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,
                1)
            # should we add a parameter to show reference?
        else:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT)

        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        if self.config.egl_rendering:
            self._pybullet_client.loadPlugin('eglRendererPlugin')
        self.num_motion_repeats = num_motion_repeats
        self.reference_motions = reference_motions
        self.num_frames, self.transformed_reference_motions, self.all_actions = expert_trajectory_to_states_actions(reference_motions[0], self.num_motion_repeats)
        self.expert_state_vector = all_states_to_expert_state_vector(self.transformed_reference_motions, self.config.sim_time_step)

        self.show_reference_model_flag = show_reference_model_flag
        self.reference_quadruped = None

        self.last_frame_time = time.time()
        
        self._state = None
        self._episode_ended = False

        # reward_idx will keep track of where we are in the movement in reference_motions
        self._reward_idx = 0

    def get_sim_timestep(self):
        return self.config.sim_time_step

    def show_reference_model(self):
        # self.reference_quadruped = self._pybullet_client.loadURDF(
        #     URDF_FILENAME, INIT_POS, INIT_ROT) # , useFixedBase=True
        # self._pybullet_client.resetBasePositionAndOrientation(
        #     self.reference_quadruped, [0, 0, 0],
        #     [0, 0, 0, 1])
        # self._pybullet_client.resetBaseVelocity(self.reference_quadruped, [0, 0, 0],
        #                                         [0, 0, 0])
        # alpha = 0.5
        # ref_col = [1, 1, 1, alpha]

        # self._pybullet_client.changeDynamics(
        #     self.reference_quadruped, -1, linearDamping=0, angularDamping=0)

        # self._pybullet_client.setCollisionFilterGroupMask(
        #     self.reference_quadruped, -1, collisionFilterGroup=0, collisionFilterMask=0)

        # self._pybullet_client.changeDynamics(
        #     self.reference_quadruped,
        #     -1,
        #     activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
        #     self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
        #     self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

        # self._pybullet_client.changeVisualShape(
        #     self.reference_quadruped, -1, rgbaColor=ref_col)

        # num_joints = self._pybullet_client.getNumJoints(
        #     self.reference_quadruped)
        
        # self._pybullet_client.resetBaseVelocity(self.reference_quadruped, [0, 0, 0],
        #                                         [0, 0, 0])

        # # remove joint dampening
        # num_joints = self._pybullet_client.getNumJoints(self.reference_quadruped)
        # for i in range(num_joints):
        #     joint_info = self._pybullet_client.getJointInfo(self.reference_quadruped, i)
        #     self._pybullet_client.changeDynamics(joint_info[0],
        #                                          -1,
        #                                          linearDamping=0,
        #                                          angularDamping=0)

        # # this seems important, perhaps there's a default motor controller in pybullet
        # for joint_index in SIM_MOTOR_IDS:
        #     self._pybullet_client.setJointMotorControl2(
        #         bodyIndex=self.reference_quadruped,
        #         jointIndex=joint_index,
        #         controlMode=self._pybullet_client.VELOCITY_CONTROL,
        #         targetVelocity=0,
        #         force=0)

        # for j in range(num_joints):
        #     self._pybullet_client.setCollisionFilterGroupMask(
        #         self.reference_quadruped, j, collisionFilterGroup=0, collisionFilterMask=0)

        #     self._pybullet_client.changeDynamics(
        #         self.reference_quadruped,
        #         j,
        #         activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
        #         self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
        #         self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

        #     self._pybullet_client.changeVisualShape(
        #         self.reference_quadruped, j, rgbaColor=ref_col)
        init_pos = np.copy(INIT_POS)
        init_pos[1] = -0.3
        init_pos[2] += 0.3
        self._ref_robot = Robot(self._pybullet_client, action_repeat=1, init_pos=init_pos)
        self._ref_robot.reset()
            
    def set_pose(self, robot, pose):
        self._pybullet_client.resetBasePositionAndOrientation(robot, pose[:POS_SIZE], pose[POS_SIZE:POS_SIZE+ROT_SIZE])
        angles = pose[POS_SIZE + ROT_SIZE:]
        for i in range(len(angles)):
            j_id = SIM_MOTOR_IDS[i]
            angle = angles[i]
            self._pybullet_client.resetJointStateMultiDof(robot, j_id, [angle], np.zeros((1,)))
        return

    def set_ref_model_pose(self, pose: Pose):
        full_pose = np.hstack([pose.root_position, pose.root_orientation, pose.joint_angles])
        self.set_pose(self._ref_robot.quadruped, full_pose)
        self.update_camera_and_sleep()
                

    def reset_me(self):
        self._state = 0
        self._episode_ended = False
        self._reward_idx = 0
        if self.render_flag:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, self.config.enable_rendering_gui)
            self._pybullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,
                1)

        # Clear the simulation world and rebuild the robot interface.
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(
            numSolverIterations=self.num_bullet_solver_iterations)
        self._pybullet_client.setTimeStep(self.config.sim_time_step)
        self._pybullet_client.setGravity(0, 0, -10)

        # Rebuild the world.
        self.ground = self._pybullet_client.loadURDF(GROUND_URDF_FILENAME)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self.env_step_counter = 0

        self._pybullet_client.resetDebugVisualizerCamera(self.config.camera_distance,
                                                         self.config.camera_yaw,
                                                         self.config.camera_pitch,
                                                         [0, 0, 0])

        if self.render_flag:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, self.config.enable_rendering_gui)
            self._pybullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,
                1)
        # Rebuild the robot
        self._robot = Robot(self._pybullet_client, action_repeat=1)
        self._robot.reset()

        if self.show_reference_model_flag:
            self.show_reference_model()

        self.last_frame_time = time.time()
        return self.get_observation()

    def get_observation(self):
        return self._robot.get_observation()
    
    def update_camera_and_sleep(self):
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        time_to_sleep = self.config.sim_time_step - time_spent
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        obs = self.get_observation()
        base_pos = obs.base_position

        # Also keep the previous orientation of the camera set by the user.
        [yaw, pitch,
            dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
        self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
                                                            base_pos)
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    
    def calculate_reward(self, reward_idx: int, observation_vector):
        expert_idx = reward_idx % self.expert_state_vector.shape[0]
        goal_row = self.expert_state_vector[expert_idx]
        return -np.linalg.norm(goal_row - observation_vector)
    
    def _step(self, action):
        if len(action.shape) > 1:
            action = action[0]
        if self.render_flag:
            self.update_camera_and_sleep()

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self._reward_idx == self.transformed_reference_motions.shape[0]:
            self._episode_ended = True

        if self._episode_ended:
            obs = self.get_observation()
            obs_vect = observed_state_to_vector(obs)
            reward = self.calculate_reward(self._reward_idx, obs_vect)
            return obs_vect, reward, self.discount_factor * (self.discount_factor**self._reward_idx)
        else:
            # robot class and put the logics here.
            new_observation = self._robot.step(action)
            self._reward_idx += 1
            obs_vect = observed_state_to_vector(new_observation)
            if not new_observation.is_safe:
                self._episode_ended = True
                reward = -10
                return obs_vect, reward, self.discount_factor * (self.discount_factor**self._reward_idx)
            else:
                reward = self.calculate_reward(self._reward_idx, obs_vect)
                next_motor_idx = self._reward_idx % self.transformed_reference_motions.shape[0]
                next_target_motor_angles = self.transformed_reference_motions[next_motor_idx, 1 + POS_SIZE + ROT_SIZE:]
                new_observation.motor_angles = next_target_motor_angles
                obs_vect = observed_state_to_vector(new_observation)
                return obs_vect, reward, self.discount_factor * (self.discount_factor**self._reward_idx)

    def step_me(self, action: "np.array"):
        if self.render_flag:
            self.update_camera_and_sleep()

        # robot class and put the logics here.
        new_observation = self._robot.step(action)

        self.env_step_counter += 1
        return new_observation

    def play_motion_files_with_movements(self, repeats_per_file: int = 3):
        for frames in self.reference_motions:
            for _ in range(repeats_per_file):
                previous_observation = self.reset_me()
                for i in range(frames.shape[0]):
                    frame = frames[i]
                    other_frame = np.array([ 0.        ,  0.        ,  0.43701   , -0.00543505,  0.04502121,
        0.04979647,  0.99772935, -1.08372851,  2.47486496,  0.62891553,
        1.71533483, -1.42093884,  2.89259569,  1.0858994 ,  2.43892924])
                    pos = frame[:POS_SIZE]
                    pos[2] += 0.4
                    rot = frame[POS_SIZE:POS_SIZE + ROT_SIZE]
                    actions = frame[POS_SIZE + ROT_SIZE:]
                    pose = Pose(pos, rot, actions)

                    if self.show_reference_model_flag:
                        self.set_ref_model_pose(pose)

                    # print(actions)
                    new_observation = self.step_me(
                        actions)
                    time.sleep(0.1)
                    # env._pybullet_client.stepSimulation()


def build_env(reference_motions: List[NDArray],
              enable_rendering: bool, show_reference_motion: bool = False, sim_time_step: float = 0.005) -> SimEnv:
    if len(reference_motions) < 1:
        raise ValueError
    sim_params = SimParams()
    sim_params.enable_rendering = enable_rendering
    sim_params.allow_knee_contact = True
    sim_params.sim_time_step = sim_time_step

    env = SimEnv(sim_params, reference_motions, show_reference_motion)
    return env


def test_pid_controller(tol: float = 0.1):
    motion_files = ["/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"]
    list_of_motion_frames = load_ref_motions(motion_files)
    env = build_env(list_of_motion_frames, enable_rendering=True,
                    show_reference_motion=True)
    env.reset_me()

    for i in range(len(DEFAULT_MOTOR_ANGLES)):
        print(f"commanding {i} motor")
        motor_angles = np.copy(DEFAULT_MOTOR_ANGLES)
        motor_angles[i] += np.pi / 4

        previous_observation = env.get_observation()
        pos = INIT_POS
        rot = INIT_ROT
        pose = Pose(pos, rot, motor_angles)
        print(pose.joint_angles)
        env.set_ref_model_pose(pose)

        while True:
            diffs = np.abs(motor_angles - previous_observation.motor_angles)
            print(diffs)
            if all(diffs < tol):
                env.reset()
                break
            previous_observation, reward, done, _ = env.step_me(
                motor_angles, previous_observation)
            time.sleep(0.01)


def test_env():
    motion_files = ["/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"]
    list_of_motion_frames = load_ref_motions(motion_files, frame_duration_override=0.01)
    env = build_env(list_of_motion_frames, enable_rendering=True,
                    show_reference_motion=True)

    env.play_motion_files_with_movements(repeats_per_file=5)


if __name__ == "__main__":
    test_env()
    # test_pid_controller()
