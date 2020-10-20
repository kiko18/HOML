#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:37:23 2020

@author: basil
"""

# uncomment this when running on colab
'''
try:
    from google.colab import drive
    %tensorflow_version 2.x
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False

if COLAB:
    !sudo apt-get install -y xvfb ffmpeg x11-utils
    !pip install -q 'gym==0.10.11'
    !pip install -q 'imageio==2.4.0'
    !pip install -q PILLOW
    !pip install -q 'pyglet==1.3.2'
    !pip install -q pyvirtualdisplay
    !pip install -q tf-agents
'''

'''
Q-Learning, as we covered in the previous part, is a robust machine learning algorithm. 
Unfortunately, Q-Learning requires that the Q-table contain an entry for every possible 
state that the environment can take. 

If the environment only includes a handful of discrete state elements, then traditional 
Q-learning might be a good learning algorithm. However, if the state space is large, 
the Q-table can become prohibitively large.

Creating policies for large state spaces is a task that Deep Q-Learning Networks (DQN) 
can usually handle. Unlike a table, a neural network does not require the program to 
represent every combination of state and action. Neural networks can generalize these 
states and learn commonalities. A DQN maps the state to its input neurons and the action 
Q-values to the output neurons. The DQN effectively becomes a function that accepts state 
and suggestions an action by returning the expected reward for each of the possible 
actions. 
'''

'''
In this script, we will make use of TF-Agents to implement a DQN to solve the cart-pole 
environment. TF-Agents makes designing, implementing, and testing new RL algorithms easier
by providing well tested modular components that can be modified and extended. 
It enables fast code iteration, with functional test integration and benchmarking.

To apply DQN to this problem, you need to create the following components for TF-Agents.
1- Environment
2- Agent
3- Policies
4- Metrics and Evaluation
5- Replay Buffer
6- Data Collection
7- Training
These components are standard in most DQN implementations. 
Later, we will apply these same components to an Atari game, and after that, 
a problem of our design. This example is based on the cart-pole tutorial provided for 
TF-Agents. We begin by importing needed Python libraries.
'''
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# Set up a virtual display for rendering OpenAI gym environments.
import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()  
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

'''
Hyperparameters
Several hyperparameters must be defined. The TF-Agent example provided reasonably 
well-tuned hyperparameters for cart-pole. Later we will adapt these to an Atari game.
'''
num_iterations = 2000#20000 # How long should training run?
initial_collect_steps = 1000   # How many initial random steps, before training start, to collect initial data.
collect_steps_per_iteration = 1  # How many steps should we run each iteration to collect data from.
replay_buffer_max_length = 100000 # How much data should we store for training examples.

batch_size = 64  
learning_rate = 1e-3 

log_interval = 200  # How often should the program provide an update.
num_eval_episodes = 10 # How many episodes should the program use for each evaluation.
eval_interval = 1000 # How often should an evaluation occur.

'''
1 -Environment
--------------
TF-Agents uses OpenAI gym environments to represent the task or problem to be solved. 
Standard environments can be created in TF-Agents using tf_agents.environments suites. 
TF-Agents has suites for loading environments from sources such as the OpenAI Gym, Atari, 
and DM Control. We begin by loading the CartPole environment from the OpenAI Gym suite.
'''

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

env.reset()
#PIL.Image.fromarray(env.render())

# The environment.step method takes an action in the environment and returns a TimeStep tuple 
# containing the next observation of the environment and the reward for the action.

# The time_step_spec() method returns the specification for the TimeStep tuple. 
# Its observation attribute shows the shape of observations, the data types, and the ranges of 
# allowed values. The reward attribute shows the same details for the reward.
print('\n Observation Spec:')
print(env.time_step_spec().observation)

print('\n Reward Spec:')
print(env.time_step_spec().reward)

#The action_spec() method returns the shape, data types, and allowed values of valid actions.
print('\n Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('\n Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('\n Next time step:')
print(next_time_step)

#Usually, the program instantiates two environments: one for training and one for evaluation.
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

# The Cartpole environment, like most environments, is written in pure Python 
# and is converted to TF-Agents and TensorFlow using the TFPyEnvironment wrapper. 
# The original environment's API uses Numpy arrays. The TFPyEnvironment turns these 
# to Tensors to make it compatible with Tensorflow agents and policies.
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

'''
2 - Agent
---------
An Agent represents the algorithm used to solve an RL problem. 
TF-Agents provides standard implementations of a variety of Agents:
DQN (used in this example)
REINFORCE
DDPG
TD3
PPO
SAC.

You can only use the DQN agent in environments which have a discrete action space. 
The DQN makes use of a QNetwork, a neural network model that learns to predict 
Q-Values (expected returns) for all actions, given a state from the environment.

The following code uses tf_agents.networks.q_network to create a QNetwork, 
passing in the observation_spec, action_spec, and a tuple describing the number 
and size of the model's hidden layers.
'''
fc_layer_params = (100,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# Now we use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. 
# In addition to the time_step_spec, action_spec and the QNetwork, 
# the agent constructor also requires an optimizer (in this case, AdamOptimizer),
#  a loss function, and an integer step counter.    
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

'''
3- Policies
-----------
A policy defines the way an agent acts in an environment. 
Typically, the goal of reinforcement learning is to train the underlying 
model until the policy produces the desired outcome.

In this example:

The desired outcome is keeping the pole balanced upright over the cart.
The policy returns an action (left or right) for each time_step observation.
Agents contain two policies:

agent.policy - The algorithm uses this main policy for evaluation and deployment.
agent.collect_policy - The algorithm this secondary policy for data collection.
'''
eval_policy = agent.policy
collect_policy = agent.collect_policy

#Policies can be created independently of agents. 
# For example, use tf_agents.policies.random_tf_policy to create a policy which will 
# randomly select an action for each time_step. We will use this random policy 
# to create initial collection data to begin training.
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# To get an action from a policy, call the policy.action(time_step) method. 
# The time_step contains the observation from the environment. 
# This method returns a PolicyStep, which is a named tuple with three components:

# action - The action to be taken (in this case, 0 or 1).
# state - Used for stateful (that is, RNN-based) policies.
# info - Auxiliary data, such as log probabilities of actions.
# Next we create an environment and setup the random policy.

example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))
time_step = example_environment.reset()
random_policy.action(time_step)

'''
4- Metrics and Evaluation
--------------------------
The most common metric used to evaluate a policy is the average return. 
The return is the sum of rewards obtained while running a policy in an environment for an episode. 
Several episodes are run, creating an average return. The following function computes 
the average return of a policy, given the policy, environment, and a number of episodes. 
We will use this same evaluation for Atari.
'''
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations 
# of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

#Running this computation on the random_policy shows a baseline performance in the environment.
compute_avg_return(eval_env, random_policy, num_eval_episodes)

'''
5- Replay Buffer
----------------
The replay buffer keeps track of data collected from the environment. 
This tutorial uses TFUniformReplayBuffer. The constructor requires the specs for 
the data it will be collecting. This value is available from the agent using the collect_data_spec method. 
The batch size and maximum buffer length are also required.
'''
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

# For most agents, collect_data_spec is a named tuple called Trajectory, 
# containing the specs for observations, actions, rewards, and other items.    
print(' \n collect_data_spec ',agent.collect_data_spec)
print(' \n collect_data_spec field ', agent.collect_data_spec._fields)

'''
6- Data Collection
------------------
Now execute the random policy in the environment for a few steps, recording 
the data in the replay buffer.
'''
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  buffer.add_batch(traj)   # Add trajectory to the replay buffer

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=100)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers

# The replay buffer is now a collection of Trajectories. 
# The agent needs access to the replay buffer. TF-Agents provides this access by creating an 
# iterable tf.data.Dataset pipeline, which will feed data to the agent.

# Each row of the replay buffer only stores a single observation step. 
# But since the DQN Agent needs both the current and next observation to compute the loss, 
# the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).

#The program also optimizes this dataset by running parallel calls and prefetching data.
# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

print(dataset)

iterator = iter(dataset)

print(iterator)

'''
7- Training the agent
---------------------
Two things must happen during the training loop:

- Collect data from the environment
- Use that data to train the agent's neural network(s)
This example also periodically evaluates the policy and prints the current score.

The following will take ~5 minutes to run.
'''

# (Optional) Optimize by wrapping some of the code in a graph 
# using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, \
                                    num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

'''
Use matplotlib.pyplot to chart how the policy improved during training.

One iteration of Cartpole-v0 consists of 200 time steps. 
The environment gives a reward of +1 for each step the pole stays up, 
so the maximum return for one episode is 200. 
The charts show the return increasing towards that maximum each time it is evaluated 
during training. (It may be a little unstable and not increase each time monotonically.)
'''
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.show()

