# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:53:17 2020

@author: BT
"""


'''
windows
-------
conda install -c conda-forge gym 
pip install --upgrade tensorflow  (if required)
pip install --user tf-agents
pip install --user gym[atari]


linux
-----
conda install -c conda-forge gym 
pip install --user tf-agents[reverb]
pip install gym[atari] or  conda install -c conda-forge gym-atari 

mac-os
------
conda install -c conda-forge gym 
pip install --user tf-agents
conda install -c conda-forge gym-atari 
'''




'''
In Reinforcement Learning, a software agent makes observations and takes actions
within an environment, and in return it receives rewards. Its objective is to learn to act
in a way that will maximize its expected rewards over time. You can think of positive 
rewards as pleasure, and negative rewards as pain. In short, the agent
acts in the environment and learns by trial and error to maximize its pleasure and
minimize its pain.

There are many examples of tasks to which Reinforcement Learning is well suited, 
such as self-driving cars, recommender systems, placing ads on a web page, or 
controlling where an image classification system should focus its attention.
'''

'''
Policy Search
-------------
The algorithm a software agent uses to determine its actions is called its policy. 
The policy could be a neural network taking observations as inputs and outputting the
action to take (see Figure 18-2).
It can also be any algorithm you can think of, and it does not have to be deterministic.
In fact, in some cases it does not even have to observe the environment! 

For example, consider a robotic vacuum cleaner whose reward is the amount of dust it
picks up in 30 minutes. Its policy could be to move forward with some probability p
every second, or randomly rotate left or right with probability 1 – p. The rotation
angle would be a random angle between –r and +r. Since this policy involves some
randomness, it is called a stochastic policy. The robot will have an erratic trajectory,
which guarantees that it will eventually get to any place it can reach and pick up all
the dust. The question is, how much dust will it pick up in 30 minutes?
How would you train such a robot? There are just two policy parameters you can
tweak: the probability p and the angle range r. 

- One possible learning algorithm could be to try out many different values for these 
parameters, and pick the combination that performs best. 
This is an example of policy search, in this case using a brute force approach. 
When the policy space is too large (which is generally the case), finding a good set 
of parameters this way is like searching for a needle in a gigantic haystack.

- Another way to explore the policy space is to use genetic algorithms. 
For example, you could randomly create a first generation of 100 policies and try them out, 
then “kill” the 80 worst policies and make the 20 survivors produce 4 offspring each.
An offspring is a copy of its parent plus some random variation. The surviving policies
plus their offspring together constitute the second generation. You can continue to
iterate through generations this way until you find a good policy.

- Yet another approach is to use optimization techniques, by evaluating the gradients of
the rewards with regard to the policy parameters, then tweaking these parameters by
following the gradients toward higher rewards.9 We will discuss this approach, is
called policy gradients (PG), in more detail later in this chapter. 

- Going back to the vacuum cleaner robot, an approach could be to slightly increase p and 
evaluate whether doing so increases the amount of dust picked up by the robot in 30 minutes; 
if it does, then increase p some more, or else reduce p. We will implement a popular PG 
algorithm using TensorFlow, but before we do, we need to create an environment for the agent
to live in—so it’s time to introduce OpenAI Gym.
'''

import gym
import time
import numpy as np
#env = gym.make('CarRacing-v0')
env = gym.make('CartPole-v1')

#for each episode we save the maximal reward. 
#This also tel us for how many step we where are to stay alive
totals = []     

for i_episode in range(10):
    obs = env.reset()   #initial observation
    episode_rewards = 0
    
    print("episode", i_episode)
    time.sleep(2)
    
    for t in range(200):
        env.render()
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print('---'*10)
        print('obs.shape=', obs.shape)
        print('reward', reward)
        print('done', done)
        print('info', info)
        episode_rewards += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
    totals.append(episode_rewards)
    
env.close()

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals)) 