# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:56:11 2020

@author: BT
"""


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

from tf_agents.environments import suite_gym
import matplotlib.pyplot as plt
import numpy as np

# create a tf_agents env
env = suite_gym.load('Breakout-v4')
print(env)   #env is just a wrapper around an OpenAI Gym environment
print(env.gym)

# TF-Agents environments are very similar to OpenAI Gym environments, but there
#are a few differences. First, the reset() method does not return an observation;
#instead it returns a TimeStep object that wraps the observation, as well as some
#extra information.

# TimeStep containt the following information:
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
action = np.array(1) #step() now expects the argument to be Numpy Array
print(env.step(action)) # # Fire


img = env.render(mode="rgb_array")

plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis("off")
plt.show()


print(env.current_time_step())


#A nice thing with TF-Agents environment is that it provides the specifications about 
#the observations, actions and time steps, including their shapes, data types, and names, 
#as well as their minimum and maximum values
print(env.observation_spec())   #screenshots of the Atari screen
print(env.action_spec())
print(env.time_step_spec())

# To know what each action corresponds to
print(env.gym.get_action_meanings())

#To render an environment, you can call env.render(mode="human"), and if you want 
#to get back the image in the form of a NumPy array, just call env.render(mode="rgb_array") 
#(unlike in OpenAI Gym, this is the default mode)

'''
Environment Wrappers and Atari Preprocessing
--------------------------------------------
The tf_agents.environments.wrappers package provides several environment wrappers. 
The role of an environment wrapper is to wrap an environment, doing so, it forward
every call to the wrapped env but can also add some  some extra functionality to it.

An example of env wrapper is the ActionDiscretizeWrapper, which Quantizes a continuous 
action space to a discrete action space. 

For example, if the original environment’s action space is the continuous range 
from –1.0 to +1.0, but you want to use an algorithm that only supports discrete 
action spaces, such as a DQN, then you can wrap the environment using 
discrete_env = ActionDiscretizeWrapper(env, num_actions=5), 
and the new discrete_env will have a discrete action space with five possible actions: 
0, 1, 2, 3, 4. These actions correspond to the actions –1.0, –0.5, 0.0, 0.5, and 1.0 
in the original environment.
'''
# show a list of available wrappers
import tf_agents.environments.wrappers

for name in dir(tf_agents.environments.wrappers):
    obj = getattr(tf_agents.environments.wrappers, name)
    if hasattr(obj, "__base__") and issubclass(obj, tf_agents.environments.wrappers.PyEnvironmentBaseWrapper):
        print("{:27s} {}".format(name, obj.__doc__.split("\n")[0]))
        
# Example of Wrapping a TF-Agents environments in a TF-Agents wrapper
# 
from tf_agents.environments.wrappers import ActionRepeat
#wrap env in an ActionRepeat wrapper, which repeats each action over n steps, while accumulating the rewards.
# In many environments, this can speed up training significantly.
repeating_env = ActionRepeat(env, times=4)
print(repeating_env)
repeating_env.unwrapped

'''
OpenAI Gym has some environment wrappers of its own in the gym.wrappers package.
We can wrap a Gym environment with a Gym wrapper, then wrap the resulting environment 
with a TF-Agents wrapper.

The suite_gym.wrap_env() function will do this for you, provided you give it a Gym 
environment and a list of Gym wrappers and/or a list of TF-Agents wrappers. 

Alternatively, the suite_gym.load() function will both create the Gym environment and
wrap it for you, if you give it some wrappers. Each wrapper will be created without
any arguments, so if you want to set some arguments, you must pass a lambda. 

For example, the following code creates a Breakout environment that will run for a 
maximum of 10,000 steps during each episode, and each action will be repeated four
times.
'''

# The suite_gym.load() function can create an env and wrap it for you, both with 
# TF-Agents environment wrappers and Gym environment wrappers (the lattergym-wrapper 
# are applied first)
from functools import partial
from gym.wrappers import TimeLimit

limited_repeating_env = suite_gym.load(
    "Breakout-v4",
    gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)], #gym wrapper
    env_wrappers=[partial(ActionRepeat, times=4)],                  #tf-agents wrapper
    #gym_env_wrappers=[lambda env: TimeLimit(env, max_episode_steps=10000)],
    #env_wrappers=[lambda env: ActionRepeat(env, times=4)])
)

print(limited_repeating_env)
limited_repeating_env.unwrapped

'''
For Atari environments, some standard preprocessing steps are applied in most papers that use them. 
TF-Agents provides a handy AtariPreprocessing wrapper that implements them. 

Here is the list of preprocessing steps it supports:
    
-Grayscale and downsampling
    Observations are converted to grayscale and downsampled (by default to 84×84 pixels).

-Max pooling
    The last two frames of the game are max-pooled using a 1 × 1 filter. 
    This is to remove the flickering that occurs in some Atari games due to the limited 
    number of sprites that the Atari 2600 could display in each frame.

-Frame skipping
    The agent only gets to see every n frames of the game (by default n = 4), and its
    actions are repeated for each frame, collecting all the rewards. This effectively
    speeds up the game from the perspective of the agent, and it also speeds up train‐
    ing because rewards are less delayed.

-End on life lost
    In some games, the rewards are just based on the score, so the agent gets no
    immediate penalty for losing a life. One solution is to end the game immediately
    whenever a life is lost. There is some debate over the actual benefits of this strat‐
    egy, so it is off by default.

Since the default Atari environment already applies random frame skipping and max pooling, 
we will need to load the raw, nonskipping variant called "BreakoutNoFrameskip-v4". 
Moreover, a single frame from the Breakout game is insufficient to know the direction and 
speed of the ball, which will make it very difficult for the agent to play the game properly 
(unless it is an RNN agent, which preserves some internal state between steps). 

One way to handle this is to use an environment wrapper that will output observations 
composed of multiple frames stacked on top of each other along the channels dimension. 
This strategy is implemented by the FrameStack4 wrapper, which returns stacks of four frames. 
Let’s create the wrapped Atari environment!
'''

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4" #variant that does not implement frame skipping 

# Create an Atari Breakout environment, and wrap it to apply the default Atari preprocessing steps
env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])

print(env)

#Play a few steps just to see what happens
env.seed(42)
env.reset()
time_step = env.step(np.array(1)) # FIRE
for _ in range(4):
    time_step = env.step(np.array(3)) # LEFT


from utils import plot_observation    
# On the plotted figure, you can see that the resolution is much lower, but sufficient 
# to play the game. Moreover, frames are stacked along the channels dimension, so red 
# represents the frame from three steps ago, green is two steps ago, blue is the previous 
# frame, and pink is the current frame.
# From this single observation, the agent can see that the ball is going toward the
# lower-left corner, and that it should continue to move the paddle to the left (as it did
# in the previous steps).

plt.figure(figsize=(6, 6))
plot_observation(time_step.observation)
plt.show()

'''
Now that we have a nice Breakout environment, with all the appropriate preprocess‐
ing and TensorFlow support, we must create the DQN agent and the other components 
we will need to train it. 
Let’s look at the architecture of the system we will build.
'''
    