import os
from typing import Tuple, List, Union, Dict
import time

from robot import Observation, INIT_POS, INIT_ROT, Pose, DEFAULT_MOTOR_ANGLES
from ref_motion_utils import POS_SIZE, ROT_SIZE, load_motion_file
from env import build_env
from util import euler_from_quaternion

import matplotlib.pyplot as plt

import numpy as np
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# we will actually do movement id, phase, orientation, velocity, cur_joint_angles. later we may add target motion to this
STATE_SIZE = 16
OUT_SIZE = 8

# Initialize neural network parameters
LAYER_SIZES = (STATE_SIZE, 512, 512, OUT_SIZE)


def expert_trajectory_to_states_actions(expert_trajectory: ArrayLike, trajectory_movement_id: int,
                                        num_repeats: int = 1, sim_timestep: float = 0.005) ->Tuple[int, ArrayLike, ArrayLike]:
    """returns the number of frames before a loop, as well as an array like entry of stuff to calculate
    rolled out trajectories with.

    Args:
        expert_trajectory (ArrayLike): _description_
        num_repeats (int, optional): _description_. Defaults to 5.

    Returns:
        Tuple[Dict[float, ArrayLike], ArrayLike]: _description_
    """
    num_frames = expert_trajectory.shape[0]
    phases = np.linspace(0, 1, num_frames)
    phases = np.hstack([phases] * num_repeats)
    phases = phases.reshape((phases.shape[0], 1))
    all_frames = np.vstack([expert_trajectory] * num_repeats)
    all_frames = np.hstack([phases, all_frames])
    # get the angles, which starts after phase, pos, rotation
    num_angles = all_frames[0, 1 + POS_SIZE + ROT_SIZE:].shape[0]
    all_actions = np.vstack([all_frames[1:, 1 + POS_SIZE + ROT_SIZE:], all_frames[0, 1 + POS_SIZE + ROT_SIZE:].reshape((1, num_angles))])
    all_states = np.zeros((all_actions.shape[0], STATE_SIZE))
    all_states[:, 0] = trajectory_movement_id
    for i in range(all_frames.shape[0]):
        if i == 0:
            prev_index = all_frames.shape[0] - 1
        else:
            prev_index = i - 1
        prev_orientation = all_frames[prev_index, 1 + POS_SIZE:1+POS_SIZE + ROT_SIZE]
        orientation = all_frames[i, 1 + POS_SIZE:1+POS_SIZE + ROT_SIZE]
        prev_roll, prev_pitch, prev_yaw = euler_from_quaternion(prev_orientation[0], prev_orientation[1], prev_orientation[2], prev_orientation[3])
        roll, pitch, yaw = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        roll_rate = (roll - prev_roll) / sim_timestep
        pitch_rate = (pitch - prev_pitch) / sim_timestep
        yaw_rate = (yaw - prev_yaw) / sim_timestep
        cur_motor_angles = all_frames[i, 1 + POS_SIZE + ROT_SIZE:]
        # phase variable
        all_states[i, 1] = all_frames[i, 0]
        all_states[i, 2:5] = np.array([roll, pitch, yaw])
        all_states[i, 5:8] = np.array([roll_rate, pitch_rate, yaw_rate])
        all_states[i, 8:] = cur_motor_angles
    
    return all_states, all_actions

def observation_to_state(obs: Observation, motion_id: int, phase: float):
    state = np.zeros((STATE_SIZE))
    state[0] = motion_id
    state[1] = phase
    state[2:5] = obs.base_orientation_euler
    state[5:8] = obs.base_angular_velocity
    state[8:] = obs.motor_angles
    return state

def observed_state_to_vector(observation: Observation):
   orientation = observation.base_orientation_euler
   base_angular_vels = observation.base_angular_velocity
   joint_angles = observation.motor_angles
   return np.hstack([orientation[0:2], base_angular_vels[0:2], joint_angles])




def relu(x):
  return jnp.maximum(0, x)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2, dtype=np.float32):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m), dtype=dtype), scale * random.normal(b_key, (n,), dtype=dtype)

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale=1e-2, dtype=np.float32):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale, dtype) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict(params: List[Tuple[ArrayLike]], x: ArrayLike):
    # per-example predictions
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits

batched_predict = jax.vmap(predict, in_axes=(None, 0))

# so we run our sim steps, we get out a trajectory with the actual outputs from sim, goal outputs from expert traj
# we could try to fit probability distribution to it


def loss(params, states, actions):
  pred_actions = batched_predict(params, states)
  return jnp.linalg.norm(actions - pred_actions)

STEP_SIZE = 0.01
@jax.jit
def update(params, states, actions):
  grads = jax.grad(loss)(params, states, actions)
  return [(w - STEP_SIZE * dw, b - STEP_SIZE * db)
          for (w, b), (dw, db) in zip(params, grads)]

def train(motion_files_ids: Dict[int, str],
          seed: Union[int, None] = None,
          cache_dir: str = "/home/mz/quadruped_learning/motion_learning/bc_cache",
          sim_timestep: float = 0.005,
          frame_duration_override: float = 0.1 ) -> List[Tuple[ArrayLike]]:
    os.makedirs(cache_dir, exist_ok=True)
    num_epochs = 3000
    batch_size = 200
    if seed is None:
        seed = 1
    key = jax.random.PRNGKey(seed)
    params = init_network_params(LAYER_SIZES, key, dtype=np.float32)

    all_states = []
    all_actions = []

    for motion_id, file_path in motion_files_ids.items():

        expert_trajectory = load_motion_file(file_path, sim_timestep=sim_timestep, frame_duration_override=frame_duration_override)
        states, actions = expert_trajectory_to_states_actions(expert_trajectory=expert_trajectory,
                                                                  trajectory_movement_id=motion_id, num_repeats=1,
                                                                    sim_timestep=sim_timestep)
        all_states.append(states)
        all_actions.append(actions)
    
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)

    N = all_states.shape[0]
    # we will do behavioral cloning where we pass in a state including a phase variable and the current orientation and angular velocity
    # we will get out next target angles for motor
    # maybe we will do that across movements and add in a target command, but let's save that for round 2
    loss_values = jnp.zeros(num_epochs + 1)
    loss_values = loss_values.at[0].set(loss(params, all_states, all_actions))
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        key, key_shuffle = jax.random.split(key, 2)
        state_shuffled = jax.random.permutation(key_shuffle, all_states)
        action_shuffled = jax.random.permutation(key_shuffle, all_actions)  # re-use the random key
        k = 0
        while k < N:
            # Sample
            state_batch = state_shuffled[k:k+batch_size]
            action_batch = action_shuffled[k:k+batch_size]
            k += batch_size
            
            params = update(params, state_batch, action_batch)
        
        loss_values = loss_values.at[epoch+1].set(loss(params, state_shuffled, action_shuffled))
        epoch_time = time.time() - start_time

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Loss: {}".format(loss_values[epoch+1]))

    fig, ax = plt.subplots()
    loss_as_np_arr = np.asarray(loss_values)
    epochs = list(range(num_epochs + 1))
    ax.plot(epochs, loss_as_np_arr)
    ax.set_xlabel("epoch_number")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss per Epoch, Behavioral Cloning, alpha={STEP_SIZE}")
    fig.savefig(os.path.join(cache_dir, "bc_loss_vs_epochs.png"))

    for (i, (w, b)) in zip(range(len(params)), params):
        w_arr = np.asarray(w)
        b_arr = np.asarray(b)
        weights_file_path = os.path.join(cache_dir, f"weights_{i}_w.npy")
        np.save(weights_file_path, w_arr)
        biases_file_path = os.path.join(cache_dir, f"weights_{i}_b.npy")
        np.save(biases_file_path, b_arr)
       

def test_scenario(motions_and_ids: Dict[int, str],
                  motion_id: int,
                  cache_dir: str = "/home/mz/quadruped_learning/motion_learning/bc_cache",
                  sim_timestep: float = 0.005,
                  num_loops: int = 10,
                  frame_duration_override: float = 0.01):
    file_path = motions_and_ids[motion_id]
    expert_trajectory = load_motion_file(file_path, sim_timestep=sim_timestep, frame_duration_override=frame_duration_override)
    states, actions = expert_trajectory_to_states_actions(expert_trajectory=expert_trajectory,
                                                                  trajectory_movement_id=motion_id, num_repeats=1,
                                                                    sim_timestep=sim_timestep)
    cache_files = os.listdir(cache_dir)
    weights_files = [file for file in cache_files if file.endswith(".npy") and file.startswith("weights_")]
    params = []
    for i in range(len(weights_files)//2):
        expected_weights_name = f"weights_{i}_w.npy"
        weights_file = os.path.join(cache_dir, expected_weights_name)
        weights_arr = np.load(weights_file)
        expected_biases_file_name = f"weights_{i}_b.npy"
        bias_file = os.path.join(cache_dir, expected_biases_file_name)
        bias_arr = np.load(bias_file)
        params.append((jnp.array(weights_arr), jnp.array(bias_arr)))
    root_pos = expert_trajectory[0, :POS_SIZE]
    root_pos[2] = 0.1
    root_orientation = expert_trajectory[0, POS_SIZE:POS_SIZE+ROT_SIZE]
    default_angles = expert_trajectory[0, POS_SIZE+ROT_SIZE:]
    # default_angles = np.copy(DEFAULT_MOTOR_ANGLES)
    # default_angles = np.zeros((8))
    default_pose = Pose(root_pos, root_orientation, default_angles)
    env = build_env(reference_motions=[expert_trajectory],
              enable_rendering=True, show_reference_motion=False, sim_time_step=sim_timestep,
              default_pose=default_pose)
    pos = np.copy(INIT_POS)
    pos[2] += 0.5
    rot = np.copy(INIT_ROT)
    env.reset_me()
    observation = env.get_observation()
    phases = states[:, 1]
    cur_state = observation_to_state(observation, motion_id, phases[0])
    num_expert_commands = len(phases)
    commanded_motor_angles = np.zeros((num_loops * num_expert_commands, 8))
    observed_motor_angles = np.zeros((num_loops * num_expert_commands, 8))
    expert_motor_angles = np.zeros((num_loops * num_expert_commands, 8))
    seconds_until_fall: float = -1
    x_distance_traveled = 0
    start_time = time.time()
    for n in range(num_loops):
        for i in range(num_expert_commands):
            
            motor_angles = predict(params, cur_state)
            commanded_motor_angles[n*num_expert_commands + i, :] = motor_angles
            # true_motor_angles = states[i, 1 + 1 + len(observation.base_orientation_euler) + len(observation.base_angular_velocity):]
            true_motor_angles = expert_trajectory[i, POS_SIZE + ROT_SIZE:]
            expert_motor_angles[n*num_expert_commands + i, :] = true_motor_angles

            pose = Pose(pos, rot, true_motor_angles)
            # env.set_ref_model_pose(pose)

            observation = env.step_me(
                    motor_angles)
            observed_motor_angles[n*num_expert_commands + i, :] = observation.motor_angles
            next_phase_idx = i + 1 if i + 1 < len(phases) else 0
            cur_state = observation_to_state(observation, motion_id, phases[next_phase_idx])
            if not observation.is_safe:
                break
        if not observation.is_safe:
                break
    end_time = time.time()
    if not observation.is_safe:
        seconds_until_fall = (n*num_expert_commands + i) * env.get_sim_timestep()
    x_distance_traveled = observation.base_position[0]
    num_motors = 8
    fig, axes = plt.subplots(
                nrows=num_motors, ncols=1, sharex=True, sharey=True,
                figsize=(10,10))
    fig.suptitle(f"Behavioral Cloning Experiment")
    time_steps = np.array(range(commanded_motor_angles.shape[0]))
    for i in range(num_motors):
        if i == 0:
            labels= ["cmd", "expert", "observed"]
        else:
            labels=['_nolegend_'] * 3
        axes[i].plot(time_steps, commanded_motor_angles[:, i], color="red", label=labels[0])
        axes[i].plot(time_steps, expert_motor_angles[:, i], color="green", label=labels[1])
        axes[i].plot(time_steps, observed_motor_angles[:, i], color="blue", label=labels[2])
        axes[i].set_ylabel(f"{i}")
    fig.text(0.5, 0.04, f'timestep ({env.get_sim_timestep()} seconds)', ha='center')
    fig.text(0.02, 0.5, 'angle (rads) (per motor)', va='center', rotation='vertical')
    fig.text(0.5, 0.02, f'time_to_fall: {round(seconds_until_fall, 2)}, x_distance: {round(x_distance_traveled, 2)}, timesteps_per_cycle: {num_expert_commands}', ha='center')
    fig.legend()
    plt.show()
    fig.savefig(os.path.join(cache_dir, "bc_motor_angles_testing.png"))
    


if __name__ == "__main__":
    example_traj = False
    frame_duration_override = None
    if example_traj:
        expert_trajectory = load_motion_file("/home/mz/quadruped_learning/data_retargetted_motion/pace.txt", sim_timestep=0.005, frame_duration_override=frame_duration_override)
        num_frames, all_states, all_actions = expert_trajectory_to_states_actions(expert_trajectory=expert_trajectory, num_repeats=5)
    example_train = False
    motion_files_ids = {0: "/home/mz/quadruped_learning/data_retargetted_motion/pace.txt",
                        1: "/home/mz/quadruped_learning/data_retargetted_motion/canter.txt",
                        2: "/home/mz/quadruped_learning/data_retargetted_motion/left turn0.txt",
                        3: "/home/mz/quadruped_learning/data_retargetted_motion/right turn0.txt",
                        4: "/home/mz/quadruped_learning/data_retargetted_motion/trot.txt"}
    if example_train:
        train(motion_files_ids, frame_duration_override=frame_duration_override)
    example_test = True
    if example_test:
       test_scenario(motion_files_ids, motion_id=4, frame_duration_override=frame_duration_override)

