# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:31:58 2020

@author: BT
"""
import numpy as np
import matplotlib.pyplot as plt

# code from previous script
# we need it to know what the true Q-value are
transition_probabilities = [ # shape=[s, a, s']
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]]
rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]

# Next, we must initialize all the Q-Values to 0
#(except for the the impossible actions, for which we set the Q-Values to –∞):
Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions

# Now let’s run the Q-Value Iteration algorithm. 
# It applies Equation (18-3) repeatedly, to all Q-Values, for every state and 
# every possible action
gamma = 0.90  # the discount factor
history1 = [] 
for iteration in range(50):
    Q_prev = Q_values.copy()
    history1.append(Q_prev) 
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])

history1 = np.array(history1) 

# For example, when the agent is in state s0 and it chooses action a1, 
# the expected sum of discounted future rewards is approximately 17.0.
#print(Q_values)

# For each state, let’s look at the action that has the highest Q-Value
#print(np.argmax(Q_values, axis=1))


'''
Q-Learning
----------
Q(s,a) <---- r + γ · max Q(s′, a′)
         α            a'
Q-Learning algorithm is an adaptation of the Q-Value Iteration algorithm to the 
situation  where the transition probabilities T(s, a, s′) and the rewards R(s, a, s′) 
are initially unknown. Q-Learning works by watching an agent play (e.g., randomly) 
and gradually improving its estimates of the Q-Values. Once it has accurate Q-Value 
estimates (or close enough), then the optimal policy is choosing the action that has 
the highest Q-Value (i.e., the greedy policy).

In general we assume that the agent initially knows only the possible states and actions, 
and nothing more. The agent must then experience each state and each transition at least 
once to know the rewards, and it must experience them multiple times if it is to have a 
reasonable estimate of the transition probabilities.

To do so, the agent uses an exploration policy —for example, a purely random policy—
to explore the MDP, and as it progresses, the Q-Learning algorithm updates the estimates 
of the Q-values based on the transitions and rewards that are actually observed.

For each state-action pair (s, a), the algorithm keeps track of a running average of the
rewards r the agent gets upon leaving the state s with action a, plus the sum of 
discounted future rewards it expects to get. To estimate this sum, we take the maximum
of the Q-Value estimates for the next state s′, since we assume that the target policy
would act optimally from then on.
'''

# Let implement Q-Learning
# We will need to simulate an agent moving around in the environment, so let's define a 
# function to perform some action and get the new state and a reward.
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward

#We also need an exploration policy, which can be any policy, as long as it visits 
#every possible state many times. We will just use a random policy, since the state 
#space is very small
def exploration_policy(state):
    return np.random.choice(possible_actions[state])

# Now let's initialize the Q-Values like earlier, and run the Q-Learning algorithm:
np.random.seed(42)

Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state][actions] = 0

alpha0 = 0.05 # initial learning rate
decay = 0.005 # learning rate decay
gamma = 0.90 # discount factor
state = 0 # initial state
history2 = [] 

for iteration in range(10000):
    history2.append(Q_values.copy()) 
    action = exploration_policy(state)  #choose a random action (in possible action for that state)
    next_state, reward = step(state, action)    #perform a step based on this action 
    next_value = np.max(Q_values[next_state])   #greedy policy at the next step
    alpha = alpha0 / (1 + iteration * decay)    #decrease learning rate
    Q_values[state, action] *= 1 - alpha        #first part of Q-learning equation
    Q_values[state, action] += alpha * (reward + gamma * next_value) #second part of Q-learning equation
    state = next_state

history2 = np.array(history2) 
print('\n Q-learning = \n', Q_values)
print('\n optimal action for each state=', np.argmax(Q_values, axis=1)) # optimal action for each state)
      
Q_value_iteration = history1[-1]
print('\n Q value iteration = \n', Q_value_iteration)

# In state S0, the optimal policy is to chose action a0, which correspond to a 
# Q-value of 18.91. 
# How many times (iterations) does Q-learning takes to find this value?
# How dos it compare to the number of iteration needed by Q-value iteration?
# The plot above reveals that Q-learning need about 8,000 iterations while 
# Q-value iteration need only 20 iterations. Obviously, not knowing the transition 
# probabilities or the rewards makes finding the optimal policy significantly harder!
true_Q_value = history1[-1, 0, 0]
print('\n Q value iteration = \n', true_Q_value)

# plot the different Q-value at each iteration for (S0, a0)
# The Q-learning algo will converge to the optimal Q-Values, but it will take many 
# iterations and possibly quite a lot of hyperparameter tuning. As you can see, 
# Q-Value Iteration algorithm (left) converges very quickly, in fewer than 20 iterations, 
# while the Q-Learning algorithm (right) takes about 8,000 iterations to converge.
# Obviously, not knowing the transition probabilities or the rewards makes finding 
# the optimal policy significantly harder!
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].set_ylabel("Q-Value$(s_0, a_0)$", fontsize=14)
axes[0].set_title("Q-Value Iteration", fontsize=14)
axes[1].set_title("Q-Learning (q_value_plot)", fontsize=14)
for ax, width, history in zip(axes, (50, 10000), (history1, history2)):
    ax.plot([0, width], [true_Q_value, true_Q_value], "k--")
    ax.plot(np.arange(width), history[:, 0, 0], "b-", linewidth=2)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.axis([0, width, 0, 24])

'''
It is somewhat surprising that Q-Learning is capable of learning the optimal policy 
by just watching an agent act randomly (imagine learning to play golf when your teacher 
is a drunk monkey). Can we do better?

NB:
The Q-Learning algorithm is called an off-policy algorithm because the policy being
trained is not necessarily the one being executed: in the previous code example, the
policy being executed (the exploration policy) is completely random, while the policy
being trained will always choose the actions with the highest Q-Values. 

Conversely, the Policy Gradients algorithm is an on-policy algorithm: it explores 
the world using the policy being trained. 
'''


'''
(Extra)
Temporal Difference (TD) Learning
---------------------------------
similary to Q-Learning, which is an adaptation of the Q-Value Iteration algorithm,
temporal Difference is an adaptation of the Value Iteration algorithm (algo that 
estimates the optimal state value  in iteration) to the situation  where the transition 
probabilities T(s, a, s′) and the rewards R(s, a, s′) are initially unknown.


V_k+1(s) <-- (1 − α)V_k(s) + α (r + γ·V_k(s′))  (18-4)
α is the learning rate (e.g., 0.01).
r + γ·V_k(s′) is called the TD target.

Since a <-α- b, implies a_k+1 ← (1 – α)·a_k + α·b_k.
Equation (18-4) can be written as V(s) <--α-- r + γ · V(s′)
'''