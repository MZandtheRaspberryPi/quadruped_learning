from dataclasses import dataclass
import time
from typing import List, Callable, Union

from numpy.typing import NDArray
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from robot import Robot, MotorControlMode, Observation, URDF_FILENAME, INIT_POS, INIT_ROT, Pose, SIM_MOTOR_IDS, DEFAULT_MOTOR_ANGLES
from ref_motion_utils import load_ref_motions, POS_SIZE, ROT_SIZE

NUM_SIMULATION_ITERATION_STEPS = 300
GROUND_URDF_FILENAME = "plane_implicit.urdf"


@dataclass
class SimParams:
    """params for the pybullet sim"""
    sim_time_step: float = 0.005
    num_action_repeat: int = 33
    enable_hard_reset: bool = False
    enable_rendering: bool = False
    enable_rendering_gui: bool = True
    robot_on_track: bool = False
    camera_distance: float = 1.0
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


class SimEnv:
    def __init__(self, config: SimParams, reference_motions: List[NDArray], show_reference_model_flag: bool = False):
        """_summary_

        Args:
            config (SimParams): _description_
            task (Callable): callable to calc reward and termination condition, takes sim env as arg
        """
        self.config = config
        self.render_flag: bool = self.config.enable_rendering

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

        self.reference_motions = reference_motions
        self.show_reference_model_flag = show_reference_model_flag
        self.reference_quadruped = None

        self.last_frame_time = time.time()
        self.reset()

    def get_sim_timestep(self):
        return self.config.sim_time_step

    def show_reference_model(self):
        self.reference_quadruped = self._pybullet_client.loadURDF(
            URDF_FILENAME, INIT_POS, INIT_ROT, useFixedBase=True)
        alpha = 0.5
        ref_col = [1, 1, 1, alpha]

        self._pybullet_client.changeDynamics(
            self.reference_quadruped, -1, linearDamping=0, angularDamping=0)

        self._pybullet_client.setCollisionFilterGroupMask(
            self.reference_quadruped, -1, collisionFilterGroup=0, collisionFilterMask=0)

        self._pybullet_client.changeDynamics(
            self.reference_quadruped,
            -1,
            activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
            self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
            self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

        self._pybullet_client.changeVisualShape(
            self.reference_quadruped, -1, rgbaColor=ref_col)

        num_joints = self._pybullet_client.getNumJoints(
            self.reference_quadruped)

        for j in range(num_joints):
            self._pybullet_client.setCollisionFilterGroupMask(
                self.reference_quadruped, j, collisionFilterGroup=0, collisionFilterMask=0)

            self._pybullet_client.changeDynamics(
                self.reference_quadruped,
                j,
                activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
                self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

            self._pybullet_client.changeVisualShape(
                self.reference_quadruped, j, rgbaColor=ref_col)

    def set_ref_model_pose(self, pose: Pose):
        self._pybullet_client.resetBasePositionAndOrientation(self.reference_quadruped,
                                                              pose.root_position,
                                                              pose.root_orientation)
        for j in range(len(pose.joint_angles)):
            joint_index = SIM_MOTOR_IDS[j]
            joint_target_angle = pose.joint_angles[j]
            j_info = self._pybullet_client.getJointInfo(
                self.reference_quadruped, joint_index)
            j_state = self._pybullet_client.getJointStateMultiDof(
                self.reference_quadruped, joint_index)
            j_pose_size = len(j_state[0])
            j_vel_size = len(j_state[1])

            if (j_pose_size > 0):
                j_pose = np.array([joint_target_angle])
                j_vel = np.zeros(j_vel_size)
                self._pybullet_client.resetJointStateMultiDof(
                    self.reference_quadruped, joint_index, j_pose, j_vel)

    def reset(self):

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

    def step(self, action: "np.array", previous_observation: Observation):
        if self.render_flag:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self.last_frame_time
            self.last_frame_time = time.time()
            time_to_sleep = self.config.sim_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            base_pos = previous_observation.base_position

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch,
             dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
                                                             base_pos)
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            # alpha = 1.
            # if self._show_reference_id >= 0:
            #     alpha = self._pybullet_client.readUserDebugParameter(
            #         self._show_reference_id)

            # ref_col = [1, 1, 1, alpha]
            # if hasattr(self._task, '_ref_model'):
            #     self._pybullet_client.changeVisualShape(
            #         self._task._ref_model, -1, rgbaColor=ref_col)
            #     for l in range(self._pybullet_client.getNumJoints(self._task._ref_model)):
            #         self._pybullet_client.changeVisualShape(
            #             self._task._ref_model, l, rgbaColor=ref_col)

            # delay = self._pybullet_client.readUserDebugParameter(
            #     self._delay_id)
            # if (delay > 0):
            #     time.sleep(delay)

        # robot class and put the logics here.
        new_observation = self._robot.step(action, previous_observation)

        # for s in self.all_sensors():
        #     s.on_step(self)

        # if self._task and hasattr(self._task, 'update'):
        #     self._task.update(self)

        # reward = self._reward()
        self.env_step_counter += 1
        return new_observation

    def play_motion_files_with_movements(self, repeats_per_file: int = 3):
        for frames in self.reference_motions:
            for _ in range(repeats_per_file):
                previous_observation = self.reset()
                for i in range(frames.shape[0]):
                    frame = frames[i]
                    pos = frame[:POS_SIZE]
                    rot = frame[POS_SIZE:POS_SIZE + ROT_SIZE]
                    actions = frame[POS_SIZE + ROT_SIZE:]
                    pose = Pose(pos, rot, actions)

                    if self.show_reference_model_flag:
                        self.set_ref_model_pose(pose)

                    # print(actions)
                    new_observation, reward, done, _ = self.step(
                        actions, previous_observation)
                    time.sleep(0.1)
                    # env._pybullet_client.stepSimulation()


def build_env(reference_motions: List[NDArray],
              enable_rendering: bool, show_reference_motion: bool = False) -> SimEnv:
    if len(reference_motions) < 1:
        raise ValueError
    sim_params = SimParams()
    sim_params.enable_rendering = enable_rendering
    sim_params.allow_knee_contact = True

    env = SimEnv(sim_params, reference_motions, show_reference_motion)
    return env


def test_pid_controller(tol: float = 0.1):
    motion_files = ["/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"]
    list_of_motion_frames = load_ref_motions(motion_files)
    env = build_env(list_of_motion_frames, enable_rendering=True,
                    show_reference_motion=True)

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
            previous_observation, reward, done, _ = env.step(
                motor_angles, previous_observation)
            time.sleep(0.01)


def test_env():
    motion_files = ["/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"]
    list_of_motion_frames = load_ref_motions(motion_files)
    env = build_env(list_of_motion_frames, enable_rendering=True,
                    show_reference_motion=True)

    env.play_motion_files_with_movements(repeats_per_file=1)


if __name__ == "__main__":
    test_env()
    # test_pid_controller()
