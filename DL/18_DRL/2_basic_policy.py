# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:10:29 2020

@author: BT
"""

'''
Reinforcement learning assumes that there is an agent that is situated in an environment.
Each step, the agent takes an action, and it receives an observation and reward from the 
environment. A RL algorithm seeks to maximize some measure of the agent’s total reward, 
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
an action, the environment transite to a new state, and also returns a 
reward that indicates the consequences of the action. 
In this task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more then 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (carl horizontal position (0.0 = center), 
its velocity (positive means right), the angle of the pole (0.0 = vertical), 
and its angular velocity (positive means clockwise).
However, neural networks can solve the task purely by looking at the scene. 

For now we will first implement a basic policy: if the pole is tilting to the left, 
then push the cart to the left, and vice versa.
'''

# install gym with:  https://anaconda.org/conda-forge/gym
#see all env in: https://gym.openai.com/envs
# see cart-pole env in: https://gym.openai.com/envs/CartPole-v0/

import gym
import time
import numpy as np

#env = gym.make('CarRacing-v0')
env = gym.make('CartPole-v1')

env.seed(42)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

#for each episode we save the maximal reward. 
#This also tel us for how many step we where are to stay alive
totals = []     
best_episode_frames = []  #save frames from the best episode and render them later on
best_episode_reward = 0
n_max_episodes = 500
n_max_steps = 200

for episode in range(n_max_episodes):
    obs = env.reset()   #initial observation
    episode_rewards = 0 #reinitialise reward for each episode 
    print("episode", episode)
    #time.sleep(2)
    
    frames = []
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        #env.render()
        action = basic_policy(obs)  #observe the environment and decide what action to take
        obs, reward, done, info = env.step(action)  #take the action       
        episode_rewards += reward #In this task, rewards are +1 for every incremental timestep
        if done:
            print("Episode finished after {} steps".format(step+1))
            break  
    if episode_rewards > best_episode_reward:    #totals[-1] = last_episode_reward
        best_episode_frames = frames
        best_episode_reward = episode_rewards
        
    totals.append(episode_rewards)


env.close()

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals)) 

# see code in intro

'''
Even with 500 tries, this policy never managed to keep the pole upright for more 
than 68 consecutive steps. Not great (the task is considered as solved for 200 steps).
Let’s see if a neural network can come up with a better policy.
''' 

import PIL
import os
image_path = os.path.join("basic_policy.gif")#"images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     #duration=30,
                     loop=0)