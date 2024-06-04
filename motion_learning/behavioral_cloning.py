from typing import Tuple, List, Union, Dict

from robot import Observation
from ref_motion_utils import POS_SIZE, ROT_SIZE, load_motion_file
from env import build_env
from util import euler_from_quaternion

import numpy as np
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



"""
state will be, 
current: phase_variable, roll, pitch, roll rate, pitch rate, 8 joint angles from time step t+1, 8 joint angles from timea step t+5

output will be 8 target joint angles
"""
STATE_SIZE = 21
OUT_SIZE = 8
# our observed state will use roll, pitch, roll rate, pitch rate, 8 joint angles we arrived at
# and calculate rewards using this
OBSERVED_STATE_SIZE = 2 + 2 + 8

def expert_trajectory_to_states_actions(expert_trajectory: ArrayLike, num_repeats: int = 5) ->Tuple[int, ArrayLike, ArrayLike]:
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


def observed_state_to_vector(observation: Observation):
   orientation = observation.base_orientation_euler
   base_angular_vels = observation.base_angular_velocity
   joint_angles = observation.motor_angles
   return np.hstack([orientation[0:2], base_angular_vels[0:2], joint_angles])

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
        


def all_states_to_neural_network_input(cur_counter: int, all_frames: ArrayLike, num_frames_per_cycle: int, observation: Observation):
   nn_inputs = np.zeros((STATE_SIZE, ))
   # phase variable
   nn_inputs[0] = all_frames[cur_counter][0]
   orientation = observation.base_orientation
   roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
   nn_inputs[1:3] = np.array([roll, pitch])
   nn_inputs[3:5] = np.array([observation.base_angular_velocity[0], observation.base_angular_velocity[1]])
   t_plus_1 = (cur_counter + 1) % num_frames_per_cycle
   t_plus_5 = (cur_counter + 5) % num_frames_per_cycle
   nn_inputs[5:13] = all_frames[t_plus_1][POS_SIZE + ROT_SIZE:]
   nn_inputs[13:21] = all_frames[t_plus_5][POS_SIZE + ROT_SIZE:]
   return nn_inputs
      

# Initialize neural network parameters
LAYER_SIZES = (STATE_SIZE, 512, 512, OUT_SIZE)

def relu(x):
  return jnp.maximum(0, x)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2, dtype=np.float32):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m), dtype=dtype), scale * random.normal(b_key, (n,), dtype=dtype)

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, dtype=np.float32):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, dtype) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


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

def loss(params: List[Tuple[ArrayLike]], nn_inputs: ArrayLike, expert_states: ArrayLike):
    pass


def train(motion_file: str, seed: Union[int, None] = None) -> List[Tuple[ArrayLike]]:
    if seed is None:
        seed = 1
    key = jax.random.PRNGKey(seed)
    params = init_network_params(LAYER_SIZES, key)
    expert_trajectory = load_motion_file("/home/mz/quadruped_learning/data_retargetted_motion/pace.txt", sim_timestep=0.005, frame_duration_override=0.04)
    num_frames_per_cycle, all_states, all_actions = expert_trajectory_to_states_actions(expert_trajectory=expert_trajectory, num_repeats=5)

    env = build_env([expert_trajectory], enable_rendering=True,
                show_reference_motion=True)
    
    observed_expert_states = all_states_to_expert_state_vector(all_states, env.get_sim_timestep())

    previous_observation = env.get_observation()
    trajectories = []
    cur_traj = np.zeros((all_states.shape[0], OBSERVED_STATE_SIZE))
    for i in tqdm(range(all_states.shape[0])):
        nn_input = all_states_to_neural_network_input(i, all_states,
                                                      num_frames_per_cycle, previous_observation)
        action = predict(params, nn_input)
        new_observation = env.step(action, previous_observation)
        observed_state_vector = observed_state_to_vector(new_observation)
        cur_traj[i, :] = observed_state_vector
        if not new_observation.is_safe:
            # fix cur_traj to last seen, exit
            cur_traj[i+1:, :] = observed_state_to_vector
            break
    
    # calculate rewards, do gradient improve, do another loop with aggregated data


if __name__ == "__main__":
    example_traj = False
    if example_traj:
        expert_trajectory = load_motion_file("/home/mz/quadruped_learning/data_retargetted_motion/pace.txt", sim_timestep=0.005, frame_duration_override=0.04)
        num_frames, all_states, all_actions = expert_trajectory_to_states_actions(expert_trajectory=expert_trajectory, num_repeats=5)
    example_train = True
    if example_train:
        motion_file="/home/mz/quadruped_learning/data_retargetted_motion/pace.txt"
        train(motion_file)



# # Define estimator as a neural network
# def f_hat(x, θ):
#     Ws, bs = θ['W'], θ['b']
#     y = Ws[0]*x + bs[0]  # assumes `x` is a scalar
#     for (W, b) in zip(Ws[1:], bs[1:]):
#         y = W@jax.nn.relu(y) + b
#     return y

# # Define loss function and gradient
# loss = lambda θ, X, Y: jnp.mean((Y - jax.vmap(f_hat, in_axes=(0, None))(X, θ))**2)
# grad_loss = jax.jit(jax.grad(loss, argnums=0))  # args like `loss`, outputs like `θ`
# loss = jax.jit(loss)


# key, *keys_W = jax.random.split(key, len(hidden_dims) + 2)
# key, *keys_b = jax.random.split(key, len(hidden_dims) + 2)
# θ = {
#     'W': [
#         0.1*jax.random.normal(keys_W[0], (hidden_dims[0],)),
#         0.1*jax.random.normal(keys_W[1], (hidden_dims[1], hidden_dims[0])),
#         0.1*jax.random.normal(keys_W[2], (hidden_dims[1],)),
#     ],
#     'b': [
#         0.1*jax.random.normal(keys_b[0], (hidden_dims[0],)),
#         0.1*jax.random.normal(keys_b[1], (hidden_dims[1],)),
#         0.1*jax.random.normal(keys_b[2]),
#     ],
# }


# # Initialize gradient-based optimizer
# learning_rate = 1e-3
# init_opt, update_opt, get_params = optimizers.adam(learning_rate)
# opt_state = init_opt(θ)
# idx = 0

# @jax.jit
# def training_step(idx, opt_state, X, Y):
#     θ = get_params(opt_state)
#     grads = grad_loss(θ, X, Y)
#     opt_state = update_opt(idx, grads, opt_state)
#     return opt_state

# # Do batch stochastic gradient descent
# batch = 20
# epochs = 100
# loss_values = jnp.zeros(epochs + 1)
# loss_values = loss_values.at[0].set(loss(θ, x, y))

# for i in tqdm(range(epochs)):
#     # Shuffle the data
#     key, key_shuffle = jax.random.split(key, 2)
#     x_shuffled = jax.random.permutation(key_shuffle, x)
#     y_shuffled = jax.random.permutation(key_shuffle, y)  # re-use the random key
    
#     # Do batch gradient descent
#     k = 0
#     while k < N:
#         # Sample
#         x_batch = x_shuffled[k:k+batch]
#         y_batch = y_shuffled[k:k+batch]
#         k += batch
        
#         # Gradient step
#         opt_state = training_step(idx, opt_state, x_batch, y_batch)
#         idx += 1
        
#     # Record loss on all data at the end of the epoch
#     θ = get_params(opt_state)
#     loss_values = loss_values.at[i+1].set(loss(θ, x, y))
