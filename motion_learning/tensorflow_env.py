import copy
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
import reverb
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.policies import actor_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from env import SimEnv, build_env
from agent import ACTION_SPEC, OBSERVATION_SPEC, INPUT_TENSOR_SPEC, ACTION_SPEC, OBSERVED_STATE_SIZE, ActionNet
from robot import Robot, MotorControlMode, Observation, URDF_FILENAME, INIT_POS, INIT_ROT, Pose, SIM_MOTOR_IDS, DEFAULT_MOTOR_ANGLES, DEFAULT_ROBOT_POSE
from ref_motion_utils import load_ref_motions, POS_SIZE, ROT_SIZE
from util import euler_from_quaternion

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

class TfSimEnv(py_environment.PyEnvironment):
    def __init__(self, reference_motions: List[NDArray], show_reference_model_flag: bool = False,
                 discount_factor: float = 0.97, default_pose = DEFAULT_ROBOT_POSE, num_motion_repeats = 3,
                 sim_time_step= 0.005):
        """_summary_

        Args:
            config (SimParams): _description_
            task (Callable): callable to calc reward and termination condition, takes sim env as arg
        """
        self.num_motion_repeats = num_motion_repeats
        self.sim_time_step = sim_time_step
        self.num_frames, self.transformed_reference_motions, self.all_actions = expert_trajectory_to_states_actions(reference_motions[0], self.num_motion_repeats)
        self.expert_state_vector = all_states_to_expert_state_vector(self.transformed_reference_motions, sim_time_step)

        self._env = build_env(reference_motions, enable_rendering=True, show_reference_motion=show_reference_model_flag,
                sim_time_step=sim_time_step, default_pose = default_pose, discount_factor=discount_factor)

        self._action_spec = ACTION_SPEC
        self._observation_spec = OBSERVATION_SPEC
        self._state = None
        self._episode_ended = False

        # reward_idx will keep track of where we are in the movement in reference_motions
        self._reward_idx = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_sim_timestep(self):
        return self._env.get_sim_timestep()

    def _reset(self):
        obs = self._env.reset_me()
        self._state = 0
        self._episode_ended = False
        self._reward_idx = 0
        obs_vect = observed_state_to_vector(obs)
        return ts.restart(obs_vect)

    
    def calculate_reward(self, reward_idx: int, observation_vector):
        expert_idx = reward_idx % self.expert_state_vector.shape[0]
        goal_row = self.expert_state_vector[expert_idx]
        return -tf.norm(goal_row - observation_vector) + self._reward_idx
    
    def _step(self, action):
        if len(action.shape) > 1:
            action = action[0]

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        

        # Make sure episodes don't go on forever.
        if self._reward_idx == self.transformed_reference_motions.shape[0]:
            self._episode_ended = True

        if self._episode_ended:
            obs = self._env.get_observation()
            obs_vect = observed_state_to_vector(obs)
            reward = self.calculate_reward(self._reward_idx, obs_vect)
            return ts.termination(obs_vect, reward)
        else:
            # robot class and put the logics here.
            new_observation = self._env._robot.step(action)
            self._reward_idx += 1
            obs_vect = observed_state_to_vector(new_observation)
            
            if self._env.render_flag:
                self._env.update_camera_and_sleep()
            if not new_observation.is_safe:
                self._episode_ended = True
                reward = -10
                return ts.termination(obs_vect, reward)
            else:
                reward = self.calculate_reward(self._reward_idx, obs_vect)
                next_motor_idx = self._reward_idx % self.transformed_reference_motions.shape[0]
                next_target_motor_angles = self.transformed_reference_motions[next_motor_idx, 1 + POS_SIZE + ROT_SIZE:]
                new_observation.motor_angles = next_target_motor_angles
                obs_vect = observed_state_to_vector(new_observation)
                return ts.transition(
                    obs_vect, reward=reward, discount=self._env.discount_factor**self._reward_idx)


def main():
    motion_files = ["/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"]
    num_eval_episodes = 5
    collect_episodes_per_iteration = 2
    log_interval = 25
    eval_interval = 50
    list_of_motion_frames = load_ref_motions(motion_files)
    # env = build_env(list_of_motion_frames, enable_rendering=True,
    #                 show_reference_motion=False)
    pose = copy.copy(DEFAULT_ROBOT_POSE)
    pose.root_position[2] = 0.1
    train_env = TfSimEnv(list_of_motion_frames, show_reference_model_flag=False,
                 discount_factor=0.97, default_pose = pose)
    train_env.reset()
    # env_name = "MinitaurBulletEnv-v0"
    # train_env = env = suite_pybullet.load(env_name)
    # env.reset()
    
    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    num_iterations = int(1e6)

    initial_collect_steps = 10000 # @param {type:"integer"}
    # initial_collect_steps = 10 # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 10000 # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 1000 # @param {type:"integer"}

    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 10000 # @param {type:"integer"}

    policy_save_interval = 1000  # @param {type:"integer"}

    tempdir = "/home/mz/quadruped_learning/cache" # tempfile.gettempdir()
    os.makedirs(tempdir, exist_ok=True)

    use_gpu = True

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(train_env))
    
    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')


    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
        

    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=train_step)

        tf_agent.initialize()


    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

    table_name = 'uniform_table'
    table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)
    
    dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)
    
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)
    
    random_policy = random_py_policy.RandomPyPolicy(
        train_env.time_step_spec(), train_env.action_spec())

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)
    
    initial_collect_actor = actor.Actor(
        train_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
    initial_collect_actor.run()


    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        train_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])
    
    eval_actor = actor.Actor(
        train_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),
        )
    

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=50),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy)
    

    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)


    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
        if _ % 100000 == 0 and _ != 0:
            print("hi")
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()

if __name__ == "__main__":
    main()
    # test_env()
    # test_pid_controller()
