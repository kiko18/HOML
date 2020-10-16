# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:02:25 2020

@author: BT
"""

import numpy as np

'''
Markov chain
------------
The probability to evolve from a state s to a state s′ is fixed, and it depends only on 
the pair (s, s′), not on past states (this is why we say that the system has no memory).

Markov Decision Process:
-----------------------
Markov decision processes (first described in the 1950s by Richard Bellman)
resemble Markov chains but with a twist: at each step, an agent can choose one
of several possible actions, and the transition probabilities depend on the chosen
action. Moreover, some state transitions return some reward (positive or negative),
and the agent’s goal is to find a policy that will maximize reward over time.
However, how can it find which strategy will gain the most reward over time?

Optimal state value V*(s):
------------------------
V*(s) = max Σ T(s, a, s′)·[ R(s, a, s′) + γ·V*(s′)] for all s

It is the sum of all discounted future rewards the agent can expect on average
after it reaches a state s, assuming it acts optimally.
stated differently, the equation says that if the agent acts optimally, then the optimal 
value of the current state is equal to the reward it will get on average after taking one 
optimal action, plus the expected optimal value of all possible next states that this 
action can lead to.

Value Iteration algorithm (optimal state value (of every possible state) in iteration)
-------------------------
V_k+1 (s) <-- max Σ T(s, a, s′) · [R(s, a, s′) + γ·V_k(s′)] for all s
V_k(s) = estimated value of state s at the kth iteration of the algorithm.

The Bellman optimal equation (optimal state value equation) leads directly to an 
algorithm that can precisely estimate the optimal state value of every possible state. 

First, we initialize all the state value estimates to zero, and then iteratively update 
them using the Value Iteration algorithm.
A remarkable result is that, given enough time, these estimates are guaranteed to 
converge to the optimal state values, corresponding to the optimal policy.


Q-Value Iteration algorithm (optimal state-action values)
---------------------------
(optimal state-action values means at each state we have a set of action values)

Q_k+1(s, a) <-- Σ T(s, a, s′) [R(s, a, s′) + γ · max Q_k(s′, a′)] for all (s,a)
Q-value (optimal state-action values) of state s with action a at the kth iteration

Now that we know the optimal states values, how do we tel the agent what to do?
That is how to combine those states with action?
In oder words, how do we compute the optimal policy? 

Bellman found a very similar algorithm to estimate the optimal state-action values, 
generally called QValues (Quality Values). Once again, you start by initializing all 
the Q-Value estimates to zero, then you update them using the Q-Value Iteration algorithm.

The optimal Q-Value of the state-action pair (s, a), noted Q*(s, a), is the sum of 
discounted future rewards the agent can expect on average after it reaches the state s 
and chooses action a, but before it sees the outcome of this action, assuming it acts 
optimally after that action.

Once you have the optimal Q-Values, defining the optimal policy, noted π*(s), is trivial:
when the agent is in state s, it should choose the action with the highest Q-Value
for that state: π*(s) = argmax Q*(s, a).
Let’s apply this algorithm to the MDP represented in Figure 18-8.
'''

#Let's define some transition probabilities, rewards and possible actions. 
#For example, in state s0, if action a0 is chosen then 
#with proba 0.7 we will go to state s0 with reward +10, 
#with probability 0.3 we will go to state s1 with no reward, 
#and with never go to state s2 (so the transition probabilities are [0.7, 0.3, 0.0], 
#and the rewards are [+10, 0, 0]):

# to know the transition probability from s2 to s0 after playing action a1,
# we will look up transition_probabilities[2][1][0] (which is 0.8). 
# Similarly, to get corresponding reward, we will look up rewards[2][1][0] (which is +40).
# And to get the list of possible actions in s2, we will look up possible_actions[2] 
#(in this case, only action a1 is possible).    
transition_probabilities = [ # shape=[s, a, s']
#              a0               a1               a2           
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],    #S0
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],               #S1
        [None, [0.8, 0.1, 0.1], None]] 
                         #S2
rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]

possible_actions = [[0, 1, 2], 
                    [0, 2], 
                    [1]]

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
    for s in range(3):  #iterate over states
        for a in possible_actions[s]:   #iterate over action (possible in this state)
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])

history1 = np.array(history1) 

# For example, when the agent is in state s0 and it chooses action a1, 
# the expected sum of discounted future rewards is approximately 17.0.
print(Q_values)

# For each state, let’s look at the action that has the highest Q-Value
print("\n For each state, the optimal Qvalue is:", np.argmax(Q_values, axis=1)) 


'''
The optimal policy for this MDP, when using a discount factor of 0.90, is to choose 
action a0 when in state s0, choose action a0 when in state s1, and finally choose 
action a1 (the only possible action) when in state s2.

Interestingly, if we increase the discount factor to 0.95, the optimal policy changes: 
in state s1 the best action becomes a2 (go through the fire!). 

This makes sense because the more you value future rewards, the more you are willing 
to put up with some pain now for the promise of future bliss.
(si l'on donne plus d'importance au futur qu'au present, alors les perspectives de 
 recompenses futures compensent les douleurs immédiates.)
'''