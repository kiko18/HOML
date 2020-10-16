# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 19:56:53 2020

@author: BT
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import utils
import time

'''
Q-learning approximation and Deep Q-learning
--------------------------------------------
The main problem with Q-Learning is that it does not scale well to large (or even
medium) MDPs with many states and actions. For example, Training an agent to play 
Ms.Pac-Man (the atari game) using Q-learning is impossible. There are to many states 
(about 150 pellets that Ms.Pac-Man can eat, each of which can be present or absent, 
i.e., already eaten. which leads to = 2^150 ≈ 10^45 states + states representing 
all possible combinations of positions for all the ghosts and Ms. Pac-Man itself).
There are more state than the number of atom in our planet, so there’s absolutely 
no way of keeping track of an estimate for every single Q-Value.

Since we can not keep track of an estimate for every single Q-Value, the solution is to
approximate the Q value. We call the approximated value Q_approximate(s,a). 
To compute Q_approximate(s,a), we need a function Q_θ(s, a) that approximates the Q-Value 
of any state-action pair (s, a) using a manageable number of parameters (given by the 
parameter vector θ). This is called Approximate Q-Learning. 

For years it was recommended to use linear combinations of handcrafted features extracted 
from the state (e.g., distance of the closest ghosts, their directions, and so on) to 
estimate Q-Values, but in 2013, DeepMind showed that using deep neural networks can work 
much better, especially for complex problems, and it does not require any feature 
engineering. 

A DNN used to estimate Q-Values is called a Deep Q-Network (DQN), and using a
DQN for Approximate Q-Learning is called Deep Q-Learning.

Now, how can we train a DQN? 
Consider the approximate Q-Value computed by the DQN for a given state-action pair (s, a).
That is Q_approximate(s,a). We now need a Q_target(s, a).
Thanks to Bellman, we know we want Q_approximate(s,a) to be as close as possible to 
the reward r that we actually observe after playing action a in state s, plus the 
discounted value of playing optimally from then on, this is Q_target(s,a). 

To estimate this sum of future discounted rewards, we can simply execute the DQN on the 
next  state s′ and for all possible actions a′. We get an approximate future Q-Value for 
each possible action. 

We then pick the highest (since we assume we will be playing optimally) and discount it, 
and this gives us an estimate of the sum of future discounted rewards. 
By summing the reward r and the future discounted value estimate, we get a target Q-Value 
y(s, a) for the state-action pair (s, a).

Q_target(s, a) = r + γ · max Q_θ(s′, a′)
                          a′
With this target Q-Value, we can run a training step using any Gradient Descent algorithm.
Specifically, we generally try to minimize the squared error between the estimated
Q-Value Q(s, a) and the target Q-Value (or the Huber loss to reduce the algorithm’s 
sensitivity to large errors). And that’s all for the basic Deep Q-Learning algorithm! 
Let’s see how to implement it to solve the CartPole environment.
'''

# The first thing we need is a Deep Q-Network. In theory, you need a neural net that
# takes a state-action pair and outputs an approximate Q-Value, but in practice it’s
# much more efficient to use a neural net that takes a state and outputs one approximate
# Q-Value for each possible action. To solve the CartPole environment, we do not
#need a very complicated neural net; a couple of hidden layers will do
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

env = gym.make("CartPole-v1")
input_shape = [4] # == env.observation_space.shape #an obs is a state
n_outputs = 2 # == env.action_space.n

#  Given a state, this Deep Q-network (DQN) will estimate, for each possible action, 
# the sum of discounted future rewards it can expect after it plays that action 
# (but before it sees its outcome). This is the Q-value.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

# To select an action using this DQN, we just pick the action with the largest predicted 
# Q-value. However, to ensure that the agent explores the environment, we choose a random 
# action with probability epsilon.
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:      #sometimes choose a random action
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis]) #sometimes choose the DQN predicted one
        return np.argmax(Q_values[0])

# Instead of training the DQN based only on the latest experiences, we will store all
# experiences in a replay buffer (or replay memory), and we will sample a random 
# training batch from it at each training iteration. This helps reduce the correlations
# between the experiences in a training batch, which tremendously helps training. 
# The replay memory will contain the agent's experiences. Each experience has the tuple form 
# (obs, action, reward, next_obs, done) representing a state, the action the agent took,
#the resulting reward, the next state it reached, and finally a Boolean indicating
#whether the episode ended at that point (done).
from collections import deque #linked list, each elt points to the next one and to previous one
replay_memory = deque(maxlen=2000)

# We will need a small function to sample a random batch of experiences from the replay buffer. 
#It will return five NumPy arrays corresponding to the five experience elements.

# Function to sample experiences from the replay memory. 
# It will return 5 NumPy arrays: [obs, actions, rewards, next_obs, dones].
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

# Let’s also create a function that will play a single step using the ε-greedy policy, 
#then store the resulting experience in the replay buffer.

# Function that will use the DQN to play one step, and record its 
# experience in the replay memory.
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)      #return action with biggest Q-value
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

# Finally, let’s create one last function that will sample a batch of experiences 
# from the replay buffer and train the DQN by performing a single Gradient Descent
# step on this batch
batch_size = 32
discount_rate = 0.95
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
loss_fn = tf.keras.losses.mean_squared_error

# Function that will sample some experiences from the replay memory & perform a training step
def training_step(batch_size): 
    experiences = sample_experiences(batch_size) #sample experiences from replay memory
    states, actions, rewards, next_states, dones = experiences
    ''' compute  Q-target/response '''
    next_Q_values = model.predict(next_states)          # apply DQN to S' of each experiences
    max_next_Q_values = np.max(next_Q_values, axis=1)   #take the max Q-value (we supose the agent act optimally)
    target_Q_values = (rewards +                        #compute Q-target/response (eq 18.7)
                       (1 - dones) * discount_rate * max_next_Q_values) 
    target_Q_values = target_Q_values.reshape(-1, 1)
    ''' compute Q_approximate(s,a) '''
    mask = tf.one_hot(actions, n_outputs)   
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)  #compute gradient
    optimizer.apply_gradients(zip(grads, model.trainable_variables))    #gradient step 
    
# This was the hardest part. Now training the model is straightforward
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = [] 
best_score = 0

'''
We run 600 episodes, each for a maximum of 200 steps. At each step, we first compute
the epsilon value for the ε-greedy policy: it will go from 1 down to 0.01, linearly,
in a bit under 500 episodes. Then we call the play_one_step() function, which
will use the ε-greedy policy to pick an action, then execute it and record the experience
in the replay buffer. If the episode is done, we exit the loop. Finally, if we are past
the 50th episode, we call the training_step() function to train the model on one
batch sampled from the replay buffer. The reason we play 50 episodes without training
is to give the replay buffer some time to fill up (if we don’t wait enough, then
there will not be enough diversity in the replay buffer). And that’s it, we just 
implemented the Deep Q-Learning algorithm!
'''

print(' \n training \n')
time.sleep(2)
for episode in range(600):  #run 600 episodes
    obs = env.reset()    
    for step in range(200): #run maximum 200 step for each episodes
        epsilon = max(1 - episode / 500, 0.01) #epsilon go from 1 to 0.01 in 500 episodes
        obs, reward, done, info = play_one_step(env, obs, epsilon) #experience is saved in buffer
        if done:
            break
    rewards.append(step) 
    if step > best_score: 
        best_weights = model.get_weights() 
        best_score = step 
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
    if episode > 50:    #give the replay_buffer some times to fill up
        training_step(batch_size)

model.set_weights(best_weights)    

# plot total reward the agent got during each episode
# this is actually how many step it played during that episode
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()


env.seed(42)
state = env.reset()

frames = []

print(' \n play \n')
time.sleep(2)
totals = []
for i_episode in range(10):
    obs = env.reset()   #initial observation
    episode_rewards = 0
    
    print("episode", i_episode)
    time.sleep(2)
    
    for t in range(200):
        #env.render()
        img = env.render(mode="rgb_array")
        frames.append(img)
        
        action = epsilon_greedy_policy(state)
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

print(np.mean(totals), round(np.std(totals),2), np.min(totals), np.max(totals)) 

#utils.plot_animation(frames)
utils.saveFrames(frames,title='deepQlearning')
'''
for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)
    
env.close()
    
utils.plot_animation(frames)
'''

'''
As you can see in the figure ploted bellow, the algorithm made no apparent progress at all 
for almost 300 episodes (in part because ε was very high at the beginning), then its 
performance suddenly skyrocketed up to 200 (which is the maximum possible performance in this
environment). That’s great news: the algorithm worked fine, and it actually ran much
faster than the Policy Gradient algorithm! But wait… just a few episodes later, it forgot
everything it knew, and its performance dropped below 25! 

This is called catastrophic forgetting, and it is one of the big problems facing virtually 
all RL algorithms: as the agent explores the environment, it updates its policy, but what it 
learns in one part of the environment may break what it learned earlier in other parts of 
the environment. The experiences are quite correlated, and the learning environment
keeps changing, which is not ideal for Gradient Descent! If you increase the size of the
replay buffer, the algorithm will be less subject to this problem. Reducing the learning
rate may also help. 

But the truth is, Reinforcement Learning is hard: training is often unstable, and you may 
need to try many hyperparameter values and random seeds before you find a combination that 
works well. For example, if you try changing the number of neurons per layer in the preceding 
from 32 to 30 or 34, the performance will never go above 100 (the DQN may be more stable with 
one hidden layer instead of two).

Reinforcement Learning is notoriously difficult, largely because of the training instabilities 
and the huge sensitivity to the choice of hyperparameter values and random seeds. 
As the researcher
Andrej Karpathy put it: “[Supervised learning] wants to work. […] RL must be forced to work.” 
You will need time, patience, perseverance, and perhaps a bit of luck too. This is a major 
reason RL is not as widely adopted as regular Deep Learning (e.g., convolutional nets). 

But there are a few real-world applications, beyond AlphaGo and Atari games: for example, 
Google uses RL to optimize its datacenter costs, and it is used in some robotics applications, 
for hyperparameter tuning, and in recommender systems.

You might wonder why we didn’t plot the loss. It turns out that loss is a poor indicator
of the model’s performance. The loss might go down, yet the agent might perform
worse (e.g., this can happen when the agent gets stuck in one small region of the 
environment, and the DQN starts overfitting this region). Conversely, the loss could go
up, yet the agent might perform better (e.g., if the DQN was underestimating the QValues,
and it starts correctly increasing its predictions, the agent will likely perform
better, getting more rewards, but the loss might increase because the DQN also sets
the targets, which will be larger too).

The basic Deep Q-Learning algorithm we’ve been using so far would be too unstable
to learn to play Atari games. So how did DeepMind do it? Well, they tweaked the
algorithm!
'''
