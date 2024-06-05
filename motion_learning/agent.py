import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.trajectories import time_step as ts
import tensorflow as tf

# actions will be the command angles for the 8 motors
ACTION_SIZE = 8
# our observed state will use roll, pitch, roll rate, pitch rate, 8 joint angles we are targeting next time step
# and calculate rewards using this
OBSERVED_STATE_SIZE = 2 + 2 + 8


# ACTION_SPEC = array_spec.BoundedArraySpec(
#                         shape=(ACTION_SIZE,), dtype=np.float32, name='action')
OBSERVATION_SPEC = array_spec.ArraySpec(
    shape=(OBSERVED_STATE_SIZE,), dtype=np.float32, name='observation')

# phase_variable, roll, pitch, roll rate, pitch rate, 8 joint angles from time step t+1, 8 joint angles from timea step t+5
# NN_INPUT_SIZE = 21

INPUT_TENSOR_SPEC = tensor_spec.TensorSpec((OBSERVED_STATE_SIZE,), tf.float32)
ACTION_SPEC = tensor_spec.BoundedTensorSpec((ACTION_SIZE,), tf.float32, name='action', minimum=0, maximum=2* np.pi)


class ActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ActionNet')
    self._output_tensor_spec = output_tensor_spec

    # self._sub_layers = [
    #     tf.keras.layers.Dense(
    #         input_tensor_spec.shape.num_elements(), name="input"),
    #     tf.keras.layers.Dense(512, activation="relu", name="dense_1"),
    #     tf.keras.layers.Dense(256, activation="relu", name="dense_2"),
    #     tf.keras.layers.Dense(output_tensor_spec.shape.num_elements(), name='actions')
    # ]

    model_input = tf.keras.layers.Input(shape=(input_tensor_spec.shape.num_elements(),))
    d1 = tf.keras.layers.Dense(512, activation="relu", name="dense_1")(model_input)
    d2 = tf.keras.layers.Dense(256, activation="relu", name="dense_2")(d1)
    actions = tf.keras.layers.Dense(output_tensor_spec.shape.num_elements(), name='actions')(d2)
    self.model = tf.keras.Model(inputs=[model_input], outputs=[actions])


  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    if len(output.shape) == 1:
       output = tf.reshape(output, (1, 12))
    if len(output.shape) == 3 and output.shape[0] == 1:
       output = tf.reshape(output, (output.shape[1], 12))
    output = self.model(output)
    actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    return actions, network_state
  

if __name__ == "__main__":
    
    time_step_spec = ts.time_step_spec(INPUT_TENSOR_SPEC)


    action_net = ActionNet(INPUT_TENSOR_SPEC, ACTION_SPEC)

    my_actor_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=ACTION_SPEC,
        actor_network=action_net)
    

    batch_size = 1
    observations = tf.ones([2] + time_step_spec.observation.shape.as_list())

    time_step = ts.restart(observations, batch_size)

    action_step = my_actor_policy.action(time_step)
    print('Action:')
    print(action_step.action)
