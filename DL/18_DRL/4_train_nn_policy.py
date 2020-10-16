# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:23:12 2020

@author: BT
"""
import tensorflow as tf
import numpy as np
import gym
import utils

seed = 42
tf.keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

'''
How to train the NN?

Evaluating Actions: The Credit Assignment Problem
-------------------------------------------------
A) problem: credit assignment
-----------------------------
If we knew what the best action was at each step, we could train the neural network as
usual, by minimizing the cross entropy between the estimated probability distribution
and the target probability distribution. It would just be regular supervised learning.
However, in Reinforcement Learning the only guidance the agent gets is through
rewards, and rewards are typically sparse and delayed. so when the agent gets a reward, 
it is hard for it to know which actions should get credited (or blamed) for it.

For example, if the agent manages to balance the pole for 100 steps, how can it know which 
of the 100 actions it took were good, and which of them were bad? All it knows is that the 
pole fell after the last action, but surely this last action is not entirely responsible. 
This is called the credit assignment problem.

B) solution: action return = Evaluate action based on future reward
--------------------------
To tackle this problem, a common strategy is to evaluate an action based on the sum
of all the rewards that come after it, usually applying a discount factor γ (gamma) at
each step. This sum of discounted rewards is called the action’s return. 

If the discount factor is close to 0, then future rewards won’t count for much compared 
to immediate rewards. Conversely, if the discount factor is close to 1, then rewards far 
into the future will count almost as much as immediate rewards. Typical discount factors 
vary from 0.9 to 0.99. With a discount factor of 0.95, rewards 13 steps into the future 
count roughly for half as much as immediate rewards (since 0.95^13 ≈ 0.5), while with a 
discount factor of 0.99, rewards 69 steps into the future count for half as much as 
immediate rewards. In the CartPole environment, actions have fairly short-term effects, 
so choosing a discount factor of 0.95 seems reasonable.

C) problem with this solution: Good action can get low return if followed by bad actions
------------------------------
Of course, a good action may be followed by several bad actions that cause the pole to
fall quickly, resulting in the good action getting a low return (similarly, a good actor
may sometimes star in a terrible movie). 

D) solution: action advantage = run many episode an compare action on average
------------------------------
However, if we play the game enough times, on average good actions will get a higher 
return than bad ones. We want to estimate how much better or worse an action is, 
compared to the other possible actions, on average. This is called the action advantage. 

For this, we must run many episodes and normalize all the action returns (by subtracting 
the mean and dividing by the standard deviation). After that, we can reasonably assume that
actions with a negative advantage were bad while actions with a positive advantage were good. 
Perfect—now that we have a way to evaluate each action, we are ready to train our first agent 
using policy gradients. Let’s see how.
'''


'''
Policy Gradients (PG): How to perform gradient step?
---------------------
As discussed earlier, PG algorithms optimize the parameters of a policy by following
the gradients toward higher rewards. 
One popular class of PG algorithms, called REINFORCE algorithms, was introduced back 
in 1992 by Ronald Williams. Here is one common variant:
    
1. First, let the neural network policy play the game several times, and at each step,
    compute the gradients that would make the chosen action even more likely, but
    don’t apply these gradients yet.
    
2. Once you have run several episodes, compute each action’s advantage (using the
    method described in the previous section).

3. If an action’s advantage is positive, it means that the action was probably good,
    and you want to apply the gradients computed earlier to make the action even
    more likely to be chosen in the future. However, if the action’s advantage is negative,
    it means the action was probably bad, and you want to apply the opposite
    gradients to make this action slightly less likely in the future. The solution is simply
    to multiply each gradient vector by the corresponding action’s advantage.
    
4. Finally, compute the mean of all the resulting gradient vectors, and use it to perform
    a Gradient Descent step.
    Let’s use tf.keras to implement this algorithm. We will train the neural network policy
    we built earlier so that it learns to balance the pole on the cart. First, we need a function
    that will play one step. We will pretend for now that whatever action it takes is
    the right one so that we can compute the loss and its gradients (these gradients will
    just be saved for a while, and we will modify them later depending on how good or
    bad the action turned out to be):
'''

'''
To summarize:

To train the neural network we will need to define the target probabilities y of each action. 
If an action is good we should increase its probability, and conversely if it is bad we should 
reduce it. But how do we know whether an action is good or bad? 
The problem is that most actions have delayed effects, so when you win or lose 
points in an episode, it is not clear which actions contributed to this result. 
was it just the last action? Or the last 10? Or just one action 50 steps earlier? 
This is called the credit assignment problem.

The Policy Gradients algorithm tackles this problem by first playing multiple episodes, 
then making the actions in good episodes slightly more likely, while actions in bad 
episodes are made slightly less likely. First we play, then we go back and think 
about what we did.

Let's start by creating a function to play a single step using the model. 
We will also pretend for now that whatever action it takes is the right one, 
so we can compute the loss and its gradients (we will just save these gradients for now, 
and modify them later depending on how good or bad the action turned out to be)
'''
# If left proba is high -> y_target will be 1. How?
# If left_proba is high, then action will most likely be False (since a random number 
# uniformally sampled between 0 and 1 will probably not be greater than left_proba). 
# And False means 0 when you cast it to a number, so y_target would be equal to 1 - 0 = 1. 
# In other words, we set the target to 1, meaning we pretend that the probability 
# of going left should have been 100% (so we took the right action).
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        # compute proba of going left
        left_proba = model(obs[np.newaxis]) #reshape obs as a batch
        #sample a random float between 0-1 and check wheter it is greater than left_proba
        #action will be False with proba left_proba or True with proba 1-left_proba
        temp_rdn = tf.random.uniform([1, 1])
        action = (temp_rdn > left_proba)
        # the right side make action to be 0 (left) or 1 (right)
        # target proba of going left = 1-action
        # if action is 0 -> target proba of going left will be 1
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        # compute loss
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    # compute gradient of the loss w.r.t the model's trainable variables
    # Again, this gradient will be tweaked later, before we apply them, 
    # depending on how good or bad the action turned out to be.    
    grads = tape.gradient(loss, model.trainable_variables)
    #play the selected action 
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    # we return the new observation, the reward , wheter the episode is endet or not,
    # and the gradient
    return obs, reward, done, grads


'''
Now let's create another function that will rely on the play_one_step() function to play 
multiple episodes, returning all the rewards and gradients, for each episode and each step.
'''
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []   #each elt will be a list containing all reward for one episode
    all_grads = []  #each elt is a list containing tuple representing gradient for each trainable var
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []  #each elt is a tuple containing one gradient tensor per trainable variable
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

'''
The Policy Gradients algorithm uses the model to play the episode several times 
(e.g., 10 times), then it goes back and looks at all the rewards, discounts them 
and normalizes them.
'''
# compute discounted rewards (action return)
def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

# normalize the discounted rewards across many episodes
# You can verify that the function discount_and_normalize_rewards() does indeed
# return the normalized action advantages for each action in both episodes. 
# Notice that the first episode was much worse than the second, so its normalized 
# advantages are all negative; all actions from the first episode would be 
# considered bad, and conversely all actions from the second episode would be 
# considered good.
def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

# Say there were 3 actions, and after each action there was a reward. 
# first 10, then 0, then -50. If we use a discount factor of 80%, 
# then the 3rd action will get -50 (full credit for the last reward), 
# but the 2nd action will only get -40 (80% credit for the last reward), 
# and the 1st action will get 80% of -40 (-32) plus full credit for the first 
# reward (+10), which leads to a discounted reward of -22
discount_rewards([10, 0, -50], discount_rate=0.8)

# normalize all discounted rewards across all episodes
discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8)

'''
We are ready to run the algorithm! Let’s define the hyperparameters fisrt. 
We will run 150 training iterations, playing 10 episodes per iteration, 
and each episode will last at most 200 steps. 
We will use a discount factor of 0.95.

We also need an optimizer and the loss function. A regular Adam optimizer with
learning rate 0.01 will do just fine, and we will use the binary cross-entropy 
loss function because we are training a binary classifier (there are two possible 
actions: left or right)
'''
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95
# number of input is the size of the observation network
n_inputs = 4 # == env.observation_space.shape[0]

optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss_fn = tf.keras.losses.binary_crossentropy

tf.keras.backend.clear_session()
np.random.seed(seed)
tf.random.set_seed(seed)

# simple sequential model to define the policy network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    tf.keras.layers.Dense(1, activation="sigmoid"), #output probability of goin left
    #If there were more than two  possible actions, there would be one output neuron 
    #per action, and we would use the softmax activation function instead.
])

env = gym.make("CartPole-v1")
#env = gym.make('BreakoutNoFrameskip-v4')
env.seed(seed);

# Build and run the training loop
for iteration in range(n_iterations):
    # 1-plays the game 10 times. returns all the rewards and gradients for every episode and step
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    
    total_rewards = sum(map(sum, all_rewards))                     
    print("\rIteration: {}, mean rewards: {:.1f}".format(          
        iteration, total_rewards / n_episodes_per_update), end="") 
    
    # 2-compute each action’s normalized advantage (which in the code we call the final_reward). 
    # This provides a measure of how good or bad each action actually was, in hindsight.
    all_final_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
    all_mean_grads = []
    
    # 3-we go through each trainable variable, and for each of them we compute the weighted mean 
    # of the gradients for that variable over all episodes and all steps, weighted by the 
    #final_reward.
    for var_index in range(len(model.trainable_variables)):
        for episode_index, final_rewards in enumerate(all_final_rewards):   #episode reward
            for step, final_reward in enumerate(final_rewards):             #steps reward
                mean_grads = tf.reduce_mean([final_reward * 
                                             all_grads[episode_index][step][var_index]], axis=0)                
        all_mean_grads.append(mean_grads)
        
    # 4-Finally, we apply these mean gradients using the optimizer: the model’s trainable
    # variables will be tweaked, and hopefully the policy will be a bit better.    
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
env.close()


frames = utils.render_policy_net(model)
utils.saveFrames(frames, 'trained_nn_policy')
#utils.plot_animation(frames)

'''
And we’re done! This code will train the neural network policy, 
and it will successfully learn to balance the pole on the cart. 
The mean reward per episode will get very close to 200 (which is the maximum 
by default with this environment). Success!

Researchers try to find algorithms that work well even when the
agent initially knows nothing about the environment. However,
unless you are writing a paper, you should not hesitate to inject
prior knowledge into the agent, as it will speed up training dramatically.
For example, since you know that the pole should be as vertical
as possible, you could add negative rewards proportional to the
pole’s angle. This will make the rewards much less sparse and speed
up training. Also, if you already have a reasonably good policy (e.g.,
hardcoded), you may want to train the neural network to imitate it
before using policy gradients to improve it.

The simple policy gradients algorithm we just trained solved the CartPole task, but it
would not scale well to larger and more complex tasks. Indeed, it is highly sample
inefficient, meaning it needs to explore the game for a very long time before it can
make significant progress. This is due to the fact that it must run multiple episodes to
estimate the advantage of each action, as we have seen. However, it is the foundation
of more powerful algorithms, such as Actor-Critic algorithms (which we will discuss
briefly at the end of this chapter).
Let first look at another popular family of algorithms.
'''