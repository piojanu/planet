# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.control import discounted_return
from planet import tools


def greedy(actions, actions_num):
  """Chooses actions greedily and encodes them as an one-hot vector"""
  indices = tf.argmax(actions, axis=-1, output_type=tf.int32)
  return tf.one_hot(indices, depth=actions_num, dtype=tf.float32)


def cross_entropy_method(
        cell, objective_fn, state, obs_shape, action_shape, horizon,
        amount=1000, topk=100, iterations=10, discount=0.99,
        min_action=-1, max_action=1, discrete_action=False):
  # Embedded observation and action shapes without batch dim.
  # In Atari case `action_shape` is number of discrete actions
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  # Flatten state dict to get first element and then get envs batch size
  original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  # It multiplies state's batch size `amount` times so each candidate starts in the same
  # (batched) env. It means that we spawn `amount` candidates for each env in batch
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
  # Again, logic to get state's batch size, but this time the extended one
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  # Any candidate don't use observation at any sequence step (it's open loop simulation)
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)  # [batch, sequence, 1]
  obs = tf.zeros((extended_batch, horizon) + obs_shape)
  length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  def iteration(mean_and_stddev, _):
    mean, stddev = mean_and_stddev
    # Sample action proposals from belief for each env in batch, candidate and horizon step
    normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
    # Action shape: (envs batch size, candidates amount, horizon) + action_shape
    action = normal * stddev[:, None] + mean[:, None]
    # Reshape to extended_batch format (original_batch * amount, horizon) + action_shape
    action = tf.reshape(action, (extended_batch, horizon) + action_shape)
    if discrete_action:
      # Normalize action scores
      action = tf.nn.l2_normalize(action, axis=-1)
      # Apply greedy policy
      postproc_action = greedy(action, action_shape[0])
    else:
      # Clip action to valid range
      action = tf.clip_by_value(action, min_action, max_action)
      # Keep continuous actions
      postproc_action = action
    # Evaluate proposal actions
    (_, state), _ = tf.nn.dynamic_rnn(
        cell, (0 * obs, postproc_action, use_obs), initial_state=initial_state)
    reward = objective_fn(state)
    return_ = discounted_return.discounted_return(
        reward, length, discount)[:, 0]
    # Reshape back to (envs batch size, candidates amount) format
    return_ = tf.reshape(return_, (original_batch, amount))
    # Indices have shape (envs batch size, topk) and those are candidates indices
    # for each env in the batch!
    _, indices = tf.nn.top_k(return_, topk, sorted=False)
    # Offset each index so it matches indices of action which has `extended_batch` first dim.
    indices += tf.range(original_batch)[:, None] * amount
    # best_actions have shape indices.shape + action.shape[1:], which is
    # (envs batch size, topk, horizon) + action_shape
    best_actions = tf.gather(action, indices)
    # Calculate new belief from best actions, shape: (envs batch size, horizon) + action_shape
    mean, variance = tf.nn.moments(best_actions, 1)
    stddev = tf.sqrt(variance + 1e-6)
    return mean, stddev

  # Initialize belief over actions (zero mean and unit variance)
  mean = tf.zeros((original_batch, horizon) + action_shape)
  stddev = tf.ones((original_batch, horizon) + action_shape)
  # Run optimisation
  mean, _ = tf.scan(
      iteration, tf.range(iterations), (mean, stddev), back_prop=False)
  # Select belief at last iterations
  mean = mean[-1]
  # Take only first action, shape: (envs batch size,) + action_shape
  mean = mean[:, 0]
  if discrete_action:
    # Apply greedy policy
    return greedy(mean, action_shape[0])
  else:
    # Return continuous actions
    return mean
