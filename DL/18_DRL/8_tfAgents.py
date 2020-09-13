# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:56:11 2020

@author: BT
"""

from tf_agents.environments import suite_gym

'''
The TF-Agents library is a Reinforcement Learning library based on TensorFlow,
developed at Google and open sourced in 2018. Just like OpenAI Gym, it provides
many off-the-shelf environments (including wrappers for all OpenAI Gym environments),
plus it supports the PyBullet library (for 3D physics simulation), DeepMind’s
DM Control library (based on MuJoCo’s physics engine), and Unity’s ML-Agents
library (simulating many 3D environments). 

It also implements many RL algorithms,including REINFORCE, DQN, and DDQN, as well as 
various RL components such as efficient replay buffers and metrics. 

It is fast, scalable, easy to use, and customizable: you can create your own environments 
and neural nets, and you can customize pretty much any component. 

We will use TF-Agents to train an agent to play Breakout, the famous Atari game, 
using the DQN algorithm (you can easily switch to another algorithm if you prefer).
'''

# create a tf_agents env, which is just a wrapper around an OpenAI Gym environment
env = suite_gym.load('Breakout-v4')
print(env)
print(env.gym)

# TF-Agents environments are very similar to OpenAI Gym environments, but there
#are a few differences. First, the reset() method does not return an observation;
#instead it returns a TimeStep object that wraps the observation, as well as some
#extra information.

# Time step containt the following information:
#   - reward and observation: same as for OpenAI Gym (except the reward is represented as a numpy).
#   - step_type: is equal to 0 for the first time step in the episode, 1 for intermediate
#                time steps, and 2 for the final time step. You can call the time step’s is_last()
#                method to check whether it is the final one or not. 
#   - discount: indicates the discount factor to use at this time step. 
#               In this example it is equal to 1, so there will be no discount at all. 
#               You can define the discount factor by setting the discount parameter when 
#               loading the environment.
#At any time, you can access the environment’s current time step by calling its 
#current_time_step() method.
env.seed(42)
print(env.reset())

#The step() method returns a TimeStep object as well
action = 1
print(env.step(action))

#TF-Agents environment provides the specifications of the observations, actions, 
#and time steps, including their shapes, data types, and names, as well as
#their minimum and maximum values
print(env.observation_spec())   #screenshots of the Atari screen
print(env.action_spec())
print(env.time_step_spec())

# know what each action corresponds to
env.gym.get_action_meanings()

#To render an environment, you can call env.render(mode="human"), and if you want 
#to get back the image in the form of a NumPy array, just call env.render(mode="rgb_array") 
#(unlike in OpenAI Gym, this is the default mode)