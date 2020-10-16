# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:29:21 2020

@author: BT
"""

'''
Let’s create a neural network policy that will take an observation as input, and
output the action to be executed. More precisely, it will estimate a probability 
for each action, and then we will select an action randomly, according to the estimated 
probabilities. In the case of the CartPole environment, there are just two possible
actions (left or right), so we only need one output neuron. It will output the probability
p of action 0 (left), and of course the probability of action 1 (right) will be 1 – p.
For example, if it outputs 0.7, then we will pick action 0 with 70% probability, or
action 1 with 30% probability.
'''
#*76114
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# number of input is the size of the observation network
n_inputs = 4 # == env.observation_space.shape[0]

# simple sequential model to define the policy network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]), 
    tf.keras.layers.Dense(1, activation="sigmoid"), #output probability of goin left
    #If there were more than two  possible actions, there would be one output neuron 
    #per action, and we would use the softmax activation function instead.
])

'''
You may wonder why we are picking a random action based on the probabilities
given by the neural network, rather than just picking the action with the highest
score. This approach lets the agent find the right balance between exploring new
actions and exploiting the actions that are known to work well. Here’s an analogy:
suppose you go to a restaurant for the first time, and all the dishes look equally
appealing, so you randomly pick one. If it turns out to be good, you can increase the
probability that you’ll order it next time, but you shouldn’t increase that probability
up to 100%, or else you will never try out the other dishes, some of which may be
even better than the one you tried.

Also note that in this particular environment, the past actions and observations can
safely be ignored, since each observation contains the environment’s full state. If there
were some hidden state, then you might need to consider past actions and observations
as well. For example, if the environment only revealed the position of the cart
but not its velocity, you would have to consider not only the current observation but
also the previous observation in order to estimate the current velocity. Another example
is when the observations are noisy; in that case, you generally want to use the past
few observations to estimate the most likely current state. The CartPole problem is
thus as simple as can be; the observations are noise-free, and they contain the environment’s
full state.
'''

# Let's write a small function that will run the model to play one episode, 
#and return the frames so we can display an animation:
import gym
import time

env = gym.make('CartPole-v1')
seed=42
n_max_episodes = 10
n_max_steps = 200
env.seed(seed)
np.random.seed(seed)
    
frames = []
totals = []     
best_episode_frames = []  #save frames from the best episode and render them later on
best_episode_reward = 0

for episode in range(n_max_episodes):
    obs = env.reset()   #initial observation
    episode_rewards = 0
    print("episode", episode)
    #time.sleep(2)
    
    for t in range(n_max_steps):
        #env.render()
        frames.append(env.render(mode="rgb_array"))
        
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        
# =============================================================================
#         print('---'*10)
#         print('obs.shape=', obs.shape)
#         print('reward', reward)
#         print('done', done)
#         print('info', info)
# =============================================================================
        episode_rewards += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
        if episode_rewards > best_episode_reward:    #totals[-1] = last_episode_reward
            best_episode_frames = frames
            best_episode_reward = episode_rewards
        
    totals.append(episode_rewards)
    
env.close()

print(np.mean(totals), round(np.std(totals),2), np.min(totals), np.max(totals)) 

'''
OK, we now have a neural network policy that will take observations and output
#action probabilities. But how do we train it?
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