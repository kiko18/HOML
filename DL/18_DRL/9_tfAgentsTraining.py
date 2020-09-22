#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:37:23 2020

@author: btousside
"""
import tensorflow as tf
import numpy as np

'''
tf_agent environment, what next?
-------------------------------
Now that we have a nice Breakout environment (see above), with all the appropriate preprocessing 
and TensorFlow support, we must create the DQN agent and the other components we will need to train it. 
Let’s look at the architecture of the system we will build.
'''

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])


'''
Training Architecture
---------------------
A TF-Agents training program is usually split into two parts that run in parallel, as
you can see in Figure 18-13: on the left, a driver explores the environment using a
collect policy to choose actions, and it collects trajectories (i.e., experiences), sending
them to an observer, which saves them to a replay buffer; on the right, an agent pulls
batches of trajectories from the replay buffer and trains some networks, which the col‐
lect policy uses. In short, the left part explores the environment and collects trajecto‐
ries, while the right part learns and updates the collect policy.

We will create all these components: first the Deep Q-Network, then the DQN agent (which will 
take care of creating the collect policy), then the replay buffer and the observer to write to it, 
then a few training metrics, then the driver, and finally the dataset. 
Once we have all the components in place, we will populate the replay buffer with some initial 
trajectories, then we will run the main training loop. So, let’s start by creating the Deep Q-Network.
'''


from tf_agents.environments.tf_py_environment import TFPyEnvironment

tf_env = TFPyEnvironment(env)


''' Creating the Deep Q-Network '''
# The TF-Agents library provides many networks in the tf_agents.networks package and its subpackages. 
# We will use the tf_agents.networks.q_network.QNetwork class

from tf_agents.networks.q_network import QNetwork

preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)

conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

# This QNetwork takes an observation as input and outputs one Q-Value per action, 
# so we must give it the specifications of the observations and the actions. 
# It starts with a preprocessing layer: a simple Lambda layer that casts the observations 
# to 32-bit floats and normalizes them (the values will range from 0.0 to 1.0). 
# The observations contain unsigned bytes, which use 4 times less space than 32-bit floats, 
# which is why we did not cast the observations to 32-bit floats earlier; we want to save RAM 
# in the replay buffer. Next, the network applies three convolutional layers: the first has 
# 32 8 × 8 filters and uses a stride of 4, etc. Lastly, it applies a dense layer with 512 units, 
# followed by a dense output layer with 4 units, one per Q-Value to output (i.e.,one per action). 
# All convolutional layers and all dense layers except the output layer use the ReLU activation 
# function by default (you can change this by setting the acti vation_fn argument). 
# The output layer does not use any activation function.
# Under the hood, a QNetwork is composed of two parts: an encoding network that processes the 
#observations, followed by a dense output layer that outputs one Q-Value per action. 
# TF-Agent’s EncodingNetwork class implements a neural network architecture found in various agents 
# (see Figure 18-14). It may have one or more inputs. For example, if each observation is composed of
# some sensor data plus an image from a camera, you will have two inputs. Each input may require some 
# preprocessing steps, in which case you can specify a list of Keras layers via the preprocessing_layers 
# argument, with one preprocessing layer per input, and the network will apply each layer to the 
# corresponding input (if an input requires multiple layers of preprocessing, you can pass a whole model, 
# since a Keras model can always be used as a layer). If there are two inputs or more, you must also
# pass an extra layer via the preprocessing_combiner argument, to combine the outputs from the preprocessing 
# layers into a single output. Next, the encoding network will optionally apply a list of convolutions 
# sequentially, provided you specify their parameters via the conv_layer_params argument. 
# This must be a list composed of 3-tuples (one per convolutional layer) indicating the The TF-Agents 
# Library number of filters, the kernel size, and the stride. After these convolutional layers, the
# encoding network will optionally apply a sequence of dense layers, if you set the fc_layer_params 
# argument: it must be a list containing the number of neurons for each dense layer. Optionally, you can 
# also pass a list of dropout rates (one per dense layer) via the dropout_layer_params argument if you want 
# to apply dropout after each dense layer. The QNetwork takes the output of this encoding network and passes
#it to the dense output layer (with one unit per action).

#The QNetwork class is flexible enough to build many differentarchitectures, but you can always build 
# your own network class if you need extra flexibility: extend the tf_agents.networks.Network class and 
#implement it like a regular custom Keras layer. The tf_agents.networks.Network class is a subclass of 
# the keras.layers.Layer class that adds some functionality required by some agents, such as the possibility 
# to easily create shallow copies of the bnetwork (i.e., copying the network’s architecture, but not its
#weights). For example, the DQNAgent uses this to create a copy of the online model.

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)



''' Create the DQN Agent '''
# The TF-Agents library implements many types of agents, located in the tf_agents.agents package and 
# its subpackages. We will use the tf_agents.agents.dqn.dqn_agent.DqnAgent class
from tf_agents.agents.dqn.dqn_agent import DqnAgent

# see TF-agents issue #113
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
#                                     epsilon=0.00001, centered=True)

train_step = tf.Variable(0) # variable that will count the number of training steps.
update_period = 4 #run a training step every 4 collect steps

# we build an optimizer using the same hyperparams as in the 2015 DQN paper
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
                                                epsilon=0.00001, centered=True) 

#we create a PolynomialDecay object that will compute the ε value for the ε-greedy collect policy, 
#given the current training step (it is normally used to decay the learning rate, hence the names 
#of the arguments, but it will work just fine to decay any other value). It will go from 1.0 
#down to 0.01 (the value used during in the 2015 DQN paper) in 1 million ALE frames, which corresponds 
#to 250,000 steps, since we use frame skipping with a period of 4. 
#Moreover, we will train the agent every 4 steps (i.e., 16 ALE frames), so ε will actually decay 
#over 62,500 training steps.
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε


# We then build the DQNAgent, passing it the time step and action specs, the QNetwork to train, 
# the optimizer, the number of training steps between target model updates, the loss function to use, 
# the discount factor (gamma), the train_step variable and a function that returns the ε value 
#(it must take no argument, which is why we need a lambda to pass the train_step).
# Note that the loss function must return an error per instance, not the mean error, which is why we 
# set reduction="none".
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))

#Lastly, we initialize the agent.
agent.initialize()

''' Creating the Replay Buffer and the Corresponding Observer '''

# replay buffer

# The TF-Agents library provides various replay buffer implementations in the tf_agents.replay_buffers 
# package. Some are purely written in Python (their module names start with py_), and others are 
# written based on TensorFlow (their module names start with tf_). We will use the TFUniformReplayBuffer 
# class in the tf_agents.replay_buffers.tf_uniform_replay_buffer package. It provides a high-performance 
#implementation of a replay buffer with uniform sampling.

# When we store two consecutive trajectories, they contain two consecutive observations with four frames 
# each (since we used the FrameStack4 wrapper), and unfortunately three of the four frames in the second 
# observation are redundant (they are already present in the first observation). In other words, we are 
# using about four times more RAM than necessary. To avoid this, you can instead use a PyHashedReplayBuffer 
# from the tf_agents.replay_buffers.py_hashed_replay_buffer package: it deduplicates data in the stored 
#trajectories along the last axis of the observations.


from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    # data_spec = The specification of the data that will be saved in the replay buffer. 
    #The DQN agent knowns what the collected data will look like, and it makes the data spec
    #available via its collect_data_spec attribute, so that’s what we give the replay buffer.
    data_spec=agent.collect_data_spec,
    # batch_size = The number of trajectories that will be added at each step. 
    # In our case, it will b one, since the driver will just execute one action per step and 
    # collect one trajectory. If the environment were a batched environment, meaning an environment
    # that takes a batch of actions at each step and returns a batch of observations, then
    # the driver would have to save a batch of trajectories at each step. Since we are using a TensorFlow 
    # replay buffer, it needs to know the size of the batches it will handle (to build the computation graph). 
    # An example of a batched environment is the ParallelPyEnvironment (from the tf_agents.environments.paral
    # lel_py_environment package): it runs multiple environments in parallel in separate processes 
    # (they can be different as long as they have the same action and observation specs), and at each step 
    # it takes a batch of actions and executes them in the environments (one action per environment), 
    # then it returns all the resulting observations.

    batch_size=tf_env.batch_size,
    # max_length = The maximum size of the replay buffer. We created a large replay buffer that can
    # store one million trajectories (as was done in the 2015 DQN paper). This will require a lot of RAM.
    max_length=1000000)


# Observer:
    
# Now we can create the observer that will write the trajectories to the replay buffer. 
# An observer is just a function (or a callable object) that takes a trajectory argument, so we
# can directly use the add_method() method (bound to the replay_buffer object) as our observer.
replay_buffer_observer = replay_buffer.add_batch

# If you wanted to create your own observer, you could write any function with a trajectory argument. 
# If it must have a state, you can write a class with a __call__(self, trajectory) method. For example, 
# here is a simple observer that will increment a counter every time it is called (except when the trajectory 
# represents a boundary between two episodes, which does not count as a step), and every 100 increments 
# it displays the progress up to a given total (the carriage return \r along with end="" ensures that the 
# displayed counter remains on the same line).

# Create a simple custom observer that counts and displays the number of times it is called 
# (except when it is passed a trajectory that represents the boundary between two episodes, 
# as this does not count as a step)
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


    