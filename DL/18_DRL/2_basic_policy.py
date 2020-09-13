# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:10:29 2020

@author: BT
"""

'''
Reinforcement learning assumes that there is an agent that is situated in an environment.
Each step, the agent takes an action, and it receives an observation and reward from the 
environment. An RL algorithm seeks to maximize some measure of the agent’s total reward, 
as the agent interacts with the environment. In the RL literature, the environment is 
formalized as a partially observable Markov decision process (POMDP).
'''

'''
We will train an agent (it can be a Deep Q Learning (DQN) agent) 
on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.

The Cart-Pole is a very simple environment composed of a cart that can move left 
or right, and pole placed vertically on top of it. The agent must move the cart 
left or right to keep the pole upright.

**Task**
The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. 

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more then 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.
The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
However, neural networks can solve the task purely by looking at the
scene, so we'll use a patch of the screen centered on the cart as an
input. Because of this, our results aren't directly comparable to the
ones from the official leaderboard - our task is much harder.
Unfortunately this does slow down the training, because we have to
render all the frames.
'''

# install gym with:  https://anaconda.org/conda-forge/gym
#see all env in: https://gym.openai.com/envs
# see cart-pole env in: https://gym.openai.com/envs/CartPole-v0/

import gym
import matplotlib.pyplot as plt
import numpy as np

def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img

gym.envs.registry.all()         #list of all available env
env = gym.make('CartPole-v1')   #create a CartPole (chariot-baton) environment
env.seed(42)
obs = env.reset()   #initialize the environment
# Initializing the environment return an observation.
# Observations vary depending on the environment. In this case it is a 1D NumPy 
# array composed of 4 floats: they represent the cart's horizontal position (0.0 = center), 
# its velocity (positive means right), the angle of the pole (0.0 = vertical), 
# and its angular velocity (positive means clockwise). 
# In atary games, observation are usually 3D array representing images.
print(obs)

# display the environment
# you can pick the rendering mode (the rendering options depend on the environment).
# for example, mode="rgb_array" will return the rendered image as a NumPy array.
# we can then encapsule the rendering into a function plot_environment.
#env.render(mode="rgb_array")
plot_environment(env)
plt.show()

# how to interact with an environment? 
# Your agent select an action from an "action space" (the set of possible actions). 
# you can see what this environment's action space looks like with env.action_space.
# This will return "Discrete(2)", which means that the possible actions are integers 
# 0 and 1, representing accelerating left (0) or right (1). 
# Other environments may have additional discrete actions, or other kinds of actions 
#(e.g., continuous). 
env.action_space #see action space

# Since the pole is leaning toward the right (obs[2] > 0), let’s accelerate the cart 
# toward the right.
action = 1  # accelerate right
obs, reward, done, info = env.step(action)
print(obs)
print(reward)
print(done)

# The step() method executes the given action and returns four values:
# - obs
# - reward:
#   In this environment, you get a reward of 1.0 at every step, no matter what you do,
#   so the goal is to keep the episode running as long as possible.

# - done:
#   This value will be True when the episode is over. This will happen when the pole
#   tilts too much, or goes off the screen, or after 200 steps (in this last case, you have
#   won). After that, the environment must be reset before it can be used again.

# - info:
#   This environment-specific dictionary can provide some extra information that
#   you may find useful for debugging or for training. For example, in some games it
#   may indicate how many lives the agent has.

# Once you have finished using an environment, you should call itsclose() method 
# to free resources.
env.close()

'''
Now how can we make the poll remain upright? 
We will need to define a policy for that. This is the strategy that the agent will 
use to select an action at each step. It can use all the past actions and observations 
to decide what to do.
Let's hard code a simple strategy: if the pole is tilting to the left, 
then push the cart to the left, and vice versa.
'''

# see code in intro

'''
Even with 500 tries, this policy never managed to keep the pole upright for more than
68 consecutive steps. Not great.
Let’s see if a neural network can come up with a better policy.
''' 