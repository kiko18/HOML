#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:37:23 2020

@author: basil
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

'''
tf_agent environment, what next?
-------------------------------
Now that we have a nice Breakout environment (see above), with all the appropriate 
preprocessing and TensorFlow support, we must create the DQN agent and the other 
components we will need to train it. 
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
collect policy to choose actions, and it collects trajectories (i.e., experiences), 
sending them to an observer, which saves them to a replay buffer; on the right, 
an agent pulls batches of trajectories from the replay buffer and trains some networks, 
which the collect policy uses. In short, the left part explores the environment and 
collects trajectories, while the right part learns and updates the collect policy.

We will create all these components: first the Deep Q-Network, then the DQN agent 
(which will take care of creating the collect policy), then the replay buffer and the 
observer to write to it, then a few training metrics, then the driver, and finally the 
dataset. Once we have all the components in place, we will populate the replay buffer 
with some initial trajectories, then we will run the main training loop. 
So, let’s start by creating the Deep Q-Network.
'''

from tf_agents.environments.tf_py_environment import TFPyEnvironment

tf_env = TFPyEnvironment(env)

''' Creating the Deep Q-Network '''
# The TF-Agents library provides many networks in it tf_agents.networks package and 
# its subpackages. We will use the tf_agents.networks.q_network.QNetwork class
from tf_agents.networks.q_network import QNetwork

preprocessing_layer = tf.keras.layers.Lambda( #The QNetwork takes an observation as input
                        lambda obs: tf.cast(obs, np.float32) / 255.)  

conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    #The QNetwork takes an observation as input and outputs one Q-Value per action, 
    # so we must give it the specifications of the observations and the actions. 
    tf_env.observation_spec(),
    tf_env.action_spec(),
    # It starts with a preprocessing layer: a simple Lambda layer that casts the observations 
    # to 32-bit floats and normalizes them (the values will range from 0.0 to 1.0). 
    # The observations contain unsigned bytes, which use 4 times less space than 32-bit floats, 
    # which is why we did not cast the observations to 32-bit floats earlier; we want 
    # to save RAM in the replay buffer. 
    preprocessing_layers=preprocessing_layer,
    # Next, the network applies three convolutional layers: the first has 32 8 × 8 filters 
    # and uses a stride of 4, etc.
    conv_layer_params=conv_layer_params,
    # Lastly, it applies a dense layer with 512 units, followed by a dense output layer 
    # with 4 units, one per Q-Value to output (i.e.,one per action). 
    # All convolutional layers and all dense layers except the output layer use the ReLU 
    # activation function by default (you can change this by setting the activation_fn 
    # argument). The output layer does not use any activation function.
    fc_layer_params=fc_layer_params
    )

# Under the hood, a QNetwork is composed of two parts: an encoding network that 
# processes the observations, followed by a dense output layer that outputs one 
# Q-Value per action. 
# TF-Agent’s EncodingNetwork class implements a neural network architecture found 
# in various agents (see Figure 18-14). 

# The EncodingNetwork may have one or more inputs. For example, if each observation is 
# composed of some sensor data plus an image from a camera, you will have two inputs. 
# Each input may require some preprocessing steps, in which case you can specify a list 
# of Keras layers via the preprocessing_layers argument, with one preprocessing layer per 
# input, and the network will apply each layer to the  corresponding input (if an input 
# requires multiple layers of preprocessing, you can  pass a whole model, since a Keras 
# model can always be used as a layer). 

# If there are two inputs or more, you must also pass an extra layer via the 
# preprocessing_combiner argument, to combine the outputs from the preprocessing 
# layers into a single output. Next, the encoding network will optionally apply 
# a list of convolutions sequentially, provided you specify their parameters via 
# the conv_layer_params argument. 

# After these convolutional layers, the encoding network will optionally apply a 
# sequence of dense layers, if you set the fc_layer_params argument: it must be a 
# list containing the number of neurons for each dense layer. Optionally, you can 
# also pass a list of dropout rates (one per dense layer) via the dropout_layer_params 
# argument if you want to apply dropout after each dense layer. 

# The QNetwork takes the output of this encoding network 
# and passes it to the dense output layer (with one unit per action).

#The QNetwork class is flexible enough to build many different architectures, 
# but you can always build your own network class if you need extra flexibility: 
# extend the tf_agents.networks.Network class and implement it like a regular custom 
# Keras layer. The tf_agents.networks.Network class is a subclass of the keras.layers.Layer 
# class that adds some functionality required by some agents, such as the possibility 
# to easily create shallow copies of the bnetwork (i.e., copying the network’s 
# architecture, but not its weights). For example, the DQNAgent uses this to create a 
# copy of the online model.

''' Create the DQN Agent '''
# The TF-Agents library implements many types of agents, located in the tf_agents.agents 
# package and its subpackages. We will use tf_agents.agents.dqn.dqn_agent.DqnAgent class
from tf_agents.agents.dqn.dqn_agent import DqnAgent

# see TF-agents issue #113
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
#                                     epsilon=0.00001, centered=True)

train_step = tf.Variable(0) # variable that will count the number of training steps.
update_period = 4 #train the model every 4 steps

# we build an optimizer using the same hyperparams as in the 2015 DQN paper
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
                                                epsilon=0.00001, centered=True) 

# we create a PolynomialDecay object that will compute the ε value for the ε-greedy 
# collect policy, given the current training step (it is normally used to decay the 
# learning rate, hence the names of the arguments, but it will work just fine to decay 
# any other value). 
# It will go from 1.0 down to 0.01 (the value used during in the 2015 DQN paper) 
# in 1 million ALE frames, which corresponds to 250,000 steps, since we use 
# frame skipping with a period of 4. 
# Moreover, we will train the agent every 4 steps (i.e., 16 ALE frames), so ε will 
# actually decay over 62,500 training steps.
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0, # initial ε
            decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
            end_learning_rate=0.01) # final ε

#https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DqnAgent
# We then build the DQNAgent, passing it the timestep specs and action specs, 
# the QNetwork to train, the optimizer, the number of training steps between target 
# model updates, the loss function to use, the discount factor (gamma), 
# the train_step variable and a function that returns the ε value (it must take no 
# argument, which is why we need a lambda to pass the train_step).
# Note that the loss function must return an error per instance, not the mean error, 
# which is why we set reduction="none".
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
# The TF-Agents library provides various replay buffer implementations in the 
# tf_agents.replay_buffers package. Some are purely written in Python (their module names 
# start with py_), and others are written based on tf (their module names start with tf_).
# We will use the TFUniformReplayBuffer class in the package
# tf_agents.replay_buffers.tf_uniform_replay_buffer. It provides a high-performance 
# implementation of a replay buffer with uniform sampling.
from tf_agents.replay_buffers import tf_uniform_replay_buffer

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
    # Since we are using a TensorFlow replay buffer, it needs to know the size of the batches 
    # it will handle (to build the computation graph). 
    # An example of a batched environment is the ParallelPyEnvironment (from the 
    # tf_agents.environments.parallel_py_environment package): It runs multiple environments 
    # in parallel in separate processes (they can be different as long as they have the same 
    # action and observation specs), and at each step it takes a batch of actions and executes 
    # them in the environments (one action per environment), then it returns all the resulting 
    # observations.
    batch_size=tf_env.batch_size,
    # max_length = The maximum size of the replay buffer. We created a large replay buffer 
    # that can store one million trajectories (as was done in the 2015 DQN paper). 
    # This will require a lot of RAM.
    max_length=500000#1000000
    )

# When we store two consecutive trajectories, they contain two consecutive observations 
# with four frames each (since we used the FrameStack4 wrapper), and unfortunately three 
# of the four frames in the second observation are redundant (they are already present 
# in the first observation). In other words, we are using about four times more RAM than 
# necessary. To avoid this, you can instead use a PyHashedReplayBuffer from the 
# tf_agents.replay_buffers.py_hashed_replay_buffer package: it deduplicates data in the 
# stored trajectories along the last axis of the observations.

# Observer:
    
# Now we can create the observer that will write the trajectories to the replay buffer. 
# An observer is just a function (or a callable object) that takes a trajectory argument, 
# so we can directly use the add_method() method (bound to the replay_buffer object) 
# as our observer.
replay_buffer_observer = replay_buffer.add_batch

# If you want to create your own observer, you could write any fct with a trajectory arg. 
# If it must have a state, you can write a class with a __call__(self, trajectory) method.
 
# For example, here is a simple observer that will increment a counter every time it is 
# called (except when the trajectory represents a boundary between two episodes, which does 
# not count as a step), and every 100 increments it displays the progress up to a given 
# total (the carriage return \r along with end="" ensures that the displayed counter remains 
# on the same line).

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

''' Create some training metrics '''
# TF-Agents implements several RL metrics in the tf_agents.metrics package, 
# some of tf_agents.metrics are purely in Python and some based on TensorFlow. 
# Let’s create a few of them in order to count the number of episodes, 
# the number of steps taken, and most importantly the average return per episode 
# and the average episode length.
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

# Discounting the rewards makes sense for training or to implement a policy, as it makes 
# it possible to balance the importance of immediate rewards with future rewards. 
# However, once an episode is over, we can evaluate how good it was overalls by summing 
# the undiscounted rewards. For this reason, the AverageReturnMetric computes the sum of 
# undiscounted rewards for each episode, and it keeps track of the streaming mean of these 
# sums over all the episodes it encounters.

# At any time, you can get the value of each of the metrics by calling its result() method. 
train_metrics[0].result()
# Alternatively, you can log all metrics by calling log_metrics(train_metrics) 
from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)


''' Create the Collect Driver/pilote '''
# As we explored in Figure 18-13, a driver is an object that explores an environment
# using a given policy, collects experiences, and broadcasts them to some observers. 
# At each step, the following things happen:
#   -   The driver passes the current timestep to the collect policy, which uses this time
#       step to choose an action and returns an action step object containing the action.
#   -   The driver then passes the action to the environment, which returns the next
#       time step.
#   -   Finally, the driver creates a trajectory object to represent this transition and
#       broadcasts it to all the observers.

# There are two main driver classes: DynamicStepDriver and DynamicEpisodeDriver.
# The first one collects experiences for a given number of steps, while the second 
# collects experiences for a given number of episodes. 

# We want to collect experiences for four steps for each training iteration 
# (as was done in the 2015 DQN paper), so let’s create a DynamicStepDriver

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env, #environment to play with, , , and 
    agent.collect_policy,   #the agent’s collect policy
    # list of observers (including the replay buffer observer and the training metrics)
    observers=[replay_buffer_observer] + train_metrics, 
    # number of steps to run (collect 4 steps for each training iteration)
    num_steps=update_period
    ) 

# We could now run the driver/pilot by calling its run() method, but it’s best to warm up 
# the replay buffer with experiences collected using a purely random policy. 
# For this, we can use the RandomTFPolicy class and create a second driver that will run 
# this policy for 20,000 steps (which is equivalent to 80,000 simulator frames, as was done 
# in the 2015 DQN paper). We can use our ShowProgress observer to display the progress
from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())

init_driver = DynamicStepDriver( # second driver that will run a random policy for some steps
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames

final_time_step, final_policy_state = init_driver.run() #run the initial driver

# We’re almost ready to run the training loop! We just need one last component: the dataset.

''' Create the Dataset '''
# To sample a batch of trajectories from the replay buffer, use its get_next() method.
# This returns the batch of trajectories plus a BufferInfo object that contains the sample
# identifiers and their sampling probabilities (this may be useful for some algorithms,
# such as PER). 
# For example, the following code will sample a small batch of two trajectories (subepisodes), 
# each containing three consecutive steps. These subepisodes are shown in Figure 18-15 
# (each row contains three consecutive steps from an episode)

# Let sample a batch of 2 trajectories (subepisodes), with 3 timesteps each and display them
tf.random.set_seed(42) # chosen to show an example of trajectory at the end of an episode

trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=3)

# The trajectories object is a named tuple, with seven fields. 
# Each field contains a tensor whose first two dimensions are [2 = trajectories and 3=steps] 
print(trajectories._fields)
# so the shape of the observation field is [2, 3, 84, 84, 4] 
# that’s 2 trajectories, each with 3 steps, and each step’s observation is 84 × 84 × 4
print(trajectories.observation.shape)

# Similarly, the step_type tensor has a shape of [2, 3]: in this example, both trajectories
# contain three consecutive steps in the middle on an episode (types 1, 1, 1). 
# In the second trajectory, you can barely see the ball at the lower left of the first 
# observation, and it disappears in the next two observations, so the agent is about to lose 
# a life, but the episode will not end immediately because it still has several lives left.
print(trajectories.step_type.numpy())

#Each trajectory is a concise representation of a sequence of consecutive time steps
#and action steps, designed to avoid redundancy. How so? Well, as you can see in
#Figure 18-16, transition n is composed of time step n, action step n, and time step n+1, 
#while transition n + 1 is composed of time step n + 1, action step n + 1, and time
#step n + 2. If we just stored these two transitions directly in the replay buffer, the time
#step n + 1 would be duplicated. To avoid this duplication, the nth trajectory step
#includes only the type and observation from time step n (not its reward and discount),
#and it does not contain the observation from time step n + 1 (however, it does
#contain a copy of the next time step’s type; that’s the only duplication).

# The to_transition() function in the tf_agents.trajectories.trajectory module
# converts a batched trajectory into a list containing a batched time_step, a batched
# action_step, and a batched next_time_step. Notice that the second dimension is 2
# instead of 3, since there are t transitions between t + 1 time steps (don’t worry if
# you’re a bit confused; you’ll get the hang of it)
from tf_agents.trajectories.trajectory import to_transition
time_steps, action_steps, next_time_steps = to_transition(trajectories)
print(time_steps.observation.shape)

plt.figure(figsize=(10, 6.8))
for row in range(2):
    for col in range(3):
        plt.subplot(2, 3, row * 3 + col + 1)
        utils.plot_observation(trajectories.observation[row, col].numpy())
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
plt.show()

#For our main training loop, instead of calling the get_next() method, we will use a
# tf.data.Dataset. This way, we can benefit from the power of the Data API (e.g., 
# parallelism and prefetching). For this, we call the replay buffer’s as_dataset() 
# method

dataset = replay_buffer.as_dataset(
            #sample batches of 64 trajectories at each training step (as in the 2015 DQN paper)
            sample_batch_size=64, 
            #each trajectorie has 2 steps (i.e., 2 steps = 1 full transition, 
            # including the next step’s observation)
            num_steps=2,
            # This dataset will process 3 elements in parallel, and prefetch 3 batches.
            num_parallel_calls=3).prefetch(3)


# For on-policy algorithms such as Policy Gradients, each experience should be sampled once, 
# used from training, and then discarded. In this case, you can still use a replay buffer, 
# but instead of using a Dataset, you would call the replay buffer’s gather_all() method at 
# each training iteration to get a tensor containing all the trajectories recorded so far, 
# then use them to perform a training step, and finally clear the replay buffer by calling 
# its clear() method.

# Now that we have all the components in place, we are ready to train the model!

''' Creating the Training Loop '''
#To speed up training, we will convert the main functions to TensorFlow Functions.
# For this we will use the tf_agents.utils.common.function() function, which wraps
# tf.function(), with some extra experimental options

# Convert the main functions to TF Functions for better performance
from tf_agents.utils.common import function
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# Let’s create a small function that will run the main training loop for n_iterations
def train_agent(n_iterations):
    time_step = None
    # The function first asks the collect policy for its initial state (given the 
    # environment batch size, which is 1 in this case). Since the policy is stateless, 
    # this returns an empty tuple (so we could have written policy_state = ())
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    # Next, we create an iterator over the dataset, and we run the training loop
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        # At each iteration, we call the driver’s run() method, passing it the current 
        # time step (initially None) and the current policy state. It will run the collect 
        # policy and collect experience for four steps (as we configured earlier),
        # broadcasting the collected trajectories to the replay buffer and the metrics.
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        # Next, we sample one batch of trajectories from the dataset, and we pass it to 
        # the agent’s train() method. It returns a train_loss object which may vary 
        # depending on the type of agent.
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        # Next, we display the iteration number and the training loss
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        # Every 1,000 iterations we log all the metrics
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            
# Now you can just call train_agent() for some number of iterations, and see the agent 
# gradually learn to play Breakout!  
# This will take a lot of computing power and a lot of patience (it may take hours, or
# even days, depending on your hardware), plus you may need to run the algorithm
# several times with different random seeds to get good results, but once it’s done, 
# the agent will be superhuman (at least at Breakout). You can also try training this DQN
# agent on other Atari games: it can achieve superhuman skill at most action games,
#but it is not so good at games with long-running storylines.
# For a comparison of this algorithm’s performance on various Atari games, see figure 3 
# in DeepMind’s 2015 paper.
train_agent(n_iterations=10000)  #    


frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1))
        prev_lives = lives

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

utils.plot_animation(frames)    

# If you want to save an animated GIF to show off your agent to your friends, 
#here's one way to do it:

import PIL
import os
image_path = os.path.join("breakout.gif")#"images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)

'''
We covered many topics in this chapter: Policy Gradients, Markov chains, Markov
decision processes, Q-Learning, Approximate Q-Learning, and Deep Q-Learning and
its main variants (fixed Q-Value targets, Double DQN, Dueling DQN, and prioritized
experience replay). We discussed how to use TF-Agents to train agents at scale, and
finally we took a quick look at a few other popular algorithms. Reinforcement Learning
is a huge and exciting field, with new ideas and algorithms popping out every day,
so I hope this chapter sparked your curiosity: there is a whole world to explore!
'''