# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:53:17 2020

@author: BT
"""


'''
pip install --upgrade tensorflow  (if required)

For video
env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,force = True)

windows
-------
conda install -c conda-forge gym 
pip install --user tf-agents   or pip install tf-agents==0.3.0

pip install --user gym[atari]
pip uninstall atari-py
pip install -f https://github.com/Kojoley/atari-py/releases atari_py


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
within an environment, and in return it receives rewards. Its objective is to learn
by trial and error to act in a way that will maximize its expected rewards over time. 

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
if it does, then increase p some more, or else reduce p. 

We will implement a popular PG algorithm using TensorFlow, but before we do, we need to create 
an environment for the agent to live in—so it’s time to introduce OpenAI Gym.
'''
import gym
#print(gym.envs.registry.all()) #list of all available env
import matplotlib.pyplot as plt
import numpy as np

Headless_server = False

def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    
    if (Headless_server):
        try:
            #apt install -y xvfb 
            # #conda install -c conda-forge pyvirtualdisplay
            import pyvirtualdisplay
            display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()  
        except ImportError:
            pass

    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img

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
