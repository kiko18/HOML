#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:37:23 2020

@author: basil
"""

HEADLESS = True
'''

if HEADLESS:
  try:
      from google.colab import drive
      %tensorflow_version 2.x
      print("Note: using Google CoLab")
  except:
      print("Note: not using Google CoLab")

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
This chapter demonstrates playing Atari Pong. 
Pong is a two-dimensional sports game that simulates table tennis. 
The player controls an in-game paddle by moving it vertically across the left or right 
side of the screen. They can compete against another player controlling a second paddle on 
the opposing side. Players use the paddles to hit a ball back and forth. 
The goal is for each player to reach eleven points before the opponent; 
you earn points when one fails to return it to the other. 

For the Atari 2600 version of Pong, a computer player (controlled by the 2600) is the 
opposing player.

This section shows how to adapt TF-Agents to an Atari game. Some changes are necessary 
when compared to the pole-cart game presented earlier in this chapter. 
You can quickly adapt this example to any Atari game by simply changing the environment name. 
However, I tuned the code presented here for Pong, and it may not perform as well for other games. 
Some tuning will likely be necessary to produce a good agent for other games.

To apply DQN to this problem, you need to create the following components for TF-Agents.
1- Environment
2- Agent
3- Policies
4- Metrics and Evaluation
5- Replay Buffer
6- Data Collection
7- Training
These components are standard in most DQN implementations. 
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

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

if HEADLESS:
  # Set up a virtual display for rendering OpenAI gym environments.
  import pyvirtualdisplay
  display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

'''
Hyperparameters
The hyperparameter for this DQN are tune for the more complex Atari game.
'''
num_iterations = 250000 # How long should training run?

initial_collect_steps = 200   # How many initial random steps, before training start, to collect initial data.
collect_steps_per_iteration = 10  # How many steps should we run each iteration to collect data from.
replay_buffer_max_length = 100000 # How much data should we store for training examples.

batch_size = 64 
learning_rate = 2.5e-3 
log_interval = 5000  # How often should the program provide an update.

num_eval_episodes = 5 # How many episodes should the program use for each evaluation.
eval_interval = 25000 # How often should an evaluation occur.

'''
1 -Environment
--------------
TF-Agents uses OpenAI gym environments to represent the task or problem to be solved. 
Standard environments can be created in TF-Agents using tf_agents.environments suites. 
TF-Agents has suites for loading environments from sources such as the OpenAI Gym, Atari, 
and DM Control.

You must handle Atari environments differently than games like cart-poll. 
Atari games typically use their 2D displays as the environment state. 
AI Gym represents Atari games as either a 3D (height by width by color) state spaced based 
on their screens, or a vector representing the state of the game's computer RAM. 
To preprocess Atari games for greater computational efficiency, we generally skip several 
frames, decrease the resolution, and discard color information. The following code shows
 how we can set up an Atari environment.
'''

env_name = "BreakoutNoFrameskip-v4"#'Breakout-v4'
#env_name = 'Pong-v0'
#env_name = 'BreakoutDeterministic-v4'
#env = suite_gym.load(env_name)

# AtariPreprocessing runs 4 frames at a time, max-pooling over the last 2
# frames. We need to account for this when computing things like update
# intervals.
ATARI_FRAME_SKIP = 4

# For the  Arcade Learning Environment (ALE), the standard is to stop training 
# about 30 minutes of real-time play (i.e. 108k frames)
max_episode_frames=108000  # ALE frames 

env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_frames / ATARI_FRAME_SKIP,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    #gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
    )
#env = batched_py_environment.BatchedPyEnvironment([env])

env.reset()
img = env.render(mode="rgb_array")

plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis("off")
plt.show()

# We are now ready to load and wrap the two environments for TF-Agents. 
# The algorithm uses the first environment for evaluation, and the second to train.
train_py_env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_frames / ATARI_FRAME_SKIP,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
    )

eval_py_env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_frames / ATARI_FRAME_SKIP,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
    )

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

'''
2 - Agent
---------
''' # hidden layer of the DQN
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)] #(filters, kernel_size, stride)
fc_layer_params=[512]

# observation will be of type uint8, we need to cast and normalize it
preprocessing_layer = tf.keras.layers.Lambda( #The QNetwork takes an observation as input
                        lambda obs: tf.cast(obs, np.float32) / 255.)  

from tf_agents.networks.q_network import QNetwork

q_net = QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
    )

# Convolutional neural networks usually are made up of several alternating pairs of 
# convolution and max-pooling layers, ultimately culminating in one or more dense layers. 
# These layers are the same types as previously seen in this course. 
# The QNetwork accepts two parameters that define the convolutional neural network structure.

# The more simple of the two parameters is fc_layer_params. This parameter specifies the size of 
# each of the dense layers. A tuple specifies the size of each of the layers in a list.

# The second parameter, named conv_layer_params, is a list of convolution layers parameters, 
# where each item is a length-three tuple indicating (filters, kernel_size, stride). 
# This implementation of QNetwork supports only convolution layers. If you desire a more complex 
# convolutional neural network, you must define your variant of the QNetwork.

# The QNetwork defined here is not the agent, instead, the QNetwork is used by the DQN agent to 
# implement the actual neural network. This allows flexibility as you can set your own class if needed.

# Next, we define the optimizer. For this example, I used RMSPropOptimizer. 
# However, AdamOptimizer is another popular choice. 
# We also create the DQN agent and reference the Q-network we just created.
optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=2.5e-4,  #learning_rate,
    decay=0.95,
    #momentum=0.0,
    epsilon=0.00001,
    centered=True
    )   

# we create a PolynomialDecay object that will compute the ε value for the ε-greedy 
# collect policy, given the current training step (it is normally used to decay the 
# learning rate, hence the names of the arguments, but it will work just fine to decay 
# any other value). 
# It will go from 1.0 down to 0.01 (the value used during in the 2015 DQN paper) 
# in 1 million ALE frames, which corresponds to 250,000 steps, since we use 
# frame skipping with a period of 4. 
# Moreover, we will train the agent every 4 steps (i.e., 16 ALE frames), so ε will 
# actually decay over 62,500 training steps.
update_period = 4 #train the model every 4 steps
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0, # initial ε
            decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
            end_learning_rate=0.01) # final ε


train_step = tf.Variable(0)

observation_spec = tensor_spec.from_spec(train_env.observation_spec())
time_step_spec = ts.time_step_spec(observation_spec)

action_spec = tensor_spec.from_spec(train_env.action_spec())
target_update_period=32000  # ALE frames
update_period=16  # ALE frames
_update_period = update_period / ATARI_FRAME_SKIP
_global_step = tf.compat.v1.train.get_or_create_global_step()


agent = dqn_agent.DqnAgent(
    time_step_spec,
    action_spec,
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000, # <=> 32,000 ALE frames
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
    gamma=0.99, # discount factor
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step)
  )

'''
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(), #time_step_spec,  # train_env.time_step_spec(),
    train_env.action_spec(), # action_spec,       # train_env.action_spec(),
    q_network=q_net,        # a tf_agents.network.Network to be used by the agent
    optimizer=optimizer,    # the optimizer to use for training
    epsilon_greedy=lambda: epsilon_fn(train_step), #0.01,    # probability of choosing a random action in the default epsilon-greedy collect policy
    n_step_update=1.0,      # he number of steps to consider when computing TD error and TD loss
    target_update_tau=1.0,  # factor for soft update of the target networks
    target_update_period=(  # period for soft update of the target networks
        target_update_period / ATARI_FRAME_SKIP / _update_period),
    # td_errors_loss_fn takes as input the target and the estimated Q values and returns the loss for each element of the batch.
    # It is the loss function of the internal Q-network and measures how close the Q-network was fit to the collected data.
    # However, it did not indicate how effective the DQN is in maximizing rewards. That is why we will need some metric later on.
    td_errors_loss_fn=common.element_wise_huber_loss, 
    gamma=0.99,
    reward_scale_factor=1.0,  #ultiplicative scale for the reward
    gradient_clipping=None,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_step_counter=train_step # an optional counter to increment every time the train op is run. Defaults to the global_step
    )  
'''

#Lastly, we initialize the agent.
agent.initialize()


'''
Metrics and Evaluation
----------------------
There are many different ways to measure the effectiveness of a model trained with reinforcement learning. 
The loss function of the internal Q-network is not a good measure of the entire DQN algorithm's overall fitness. 
The network loss function measures how close the Q-network was fit to the collected data and did not indicate 
how effective the DQN is in maximizing rewards. The method used for this example tracks the average reward received 
over several episodes.
'''
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():  #one episode
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# See also the metrics module for standard implementations of 
# different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

'''
Replay Buffer
DQN works by training a neural network to predict the Q-values for every possible environment-state.
A neural network needs training data, so the algorithm accumulates this training data as it runs episodes. 
The replay buffer is where this data is stored. Only the most recent episodes are stored, 
older episode data rolls off the queue as the queue accumulates new data.
'''
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    # data_spec = The specification of the data that will be saved in the replay buffer. 
    # The DQN agent knowns what the collected data will look like, and it makes the data 
    # spec available via its collect_data_spec attribute, so that’s what we give the 
    # replay buffer.
    data_spec=agent.collect_data_spec,
    # batch_size = The number of trajectories that will be added at each step. 
    # In our case, it will be one, since the driver will just execute one action per step 
    # and collect one trajectory. If the environment were a batched environment, meaning 
    # an environment that takes a batch of actions at each step and returns a batch of 
    # observations, then the driver would have to save a batch of trajectories at each step. 
    batch_size=train_env.batch_size,
    # max_length = The maximum size of the replay buffer.
    # The larger is max_length the more RAM is required.
    # If you do not have enaugh RAM, you will get an "out of memory" error
    max_length=replay_buffer_max_length   #The maximum size of the replay buffer
    )

# Dataset generates trajectories with shape [Bx2x...]
# To sample a batch of trajectories from the replay buffer, we can either use the get_next() method,
# which returns the batch of trajectories plus a BufferInfo object that contains the sample
# identifiers and their sampling probabilities. 
# Or the as_dataset method which is a tf.data.Dataset. This way, we can benefit from the power of 
# the Data API (e.g., parallelism and prefetching).  
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2
    ).prefetch(3)

'''
Random (Data) Collection
------------------------
The algorithm must prime the pump. Training cannot begin on an empty replay buffer. 
The following code performs a predefined number of steps to generate initial training data.
'''
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

def collect_step(environment, policy, buffer):
  # generate a trajectory
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  # Add the trajectory to the replay buffer
  buffer.add_batch(traj)

# collect some data with a random policy
def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

'''
Training the agent
------------------
Now that we have pump some data to the replay buffer we'are ready to train the DQN. 
This process can take many hours, depending on how many episodes you wish to run through. 
As training occurs, loss and average return will be updated. 
As training becomes more successful, the average return should increase. 
The losses reflect the average loss for individual training batches.

Two things must happen during the training loop:
  - collect data from the environment
  - use that data to train the agent's neural network(s)
'''

# create an iterator on dataset
iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# training loop
for _ in range(num_iterations):
  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data (trajectory) from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

'''
Visualization
-------------
The notebook can plot the average return over training iterations. 
The average return should increase as the program performs more training iterations.
'''    
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=10)


def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  #with imageio.get_writer(filename, fps=fps) as video:
  video = []
  for _ in range(num_episodes):
    time_step = eval_env.reset()
    #video.append_data(eval_py_env.render())
    #video.append(eval_py_env.pyenv.envs[0].render(mode="rgb_array"))
    video.append(eval_py_env.render(mode="rgb_array"))
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_env.step(action_step.action)
      #video.append_data(eval_py_env.render())
      #video.append(eval_py_env.pyenv.envs[0].render(mode="rgb_array"))
      video.append(eval_py_env.render(mode="rgb_array"))
  return video  #embed_mp4(filename)

frames = create_policy_eval_video(agent.policy, "trained-agent")  

import PIL
import os
image_path = os.path.join("pong__eval.gif")#"images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]#[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)


image_path = os.path.join("breakout__eval.gif")#"images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)

print('\n frame length = ', len(frames))

print('done')


def create_policy_eval_video2(policy, filename, num_episodes=5, fps=30):
  video = []
  for _ in range(num_episodes):
    time_step = train_env.reset()
    #video.append_data(eval_py_env.render())
    #video.append(eval_py_env.pyenv.envs[0].render(mode="rgb_array"))
    video.append(train_py_env.render(mode="rgb_array"))
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = train_env.step(action_step.action)
      #video.append_data(eval_py_env.render())
      #video.append(eval_py_env.pyenv.envs[0].render(mode="rgb_array"))
      video.append(train_py_env.render(mode="rgb_array"))
  return video  #embed_mp4(filename)

frames = create_policy_eval_video2(agent.policy, "trained-agent")  


import PIL
import os
image_path = os.path.join("pong__train.gif")#"images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]#[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)