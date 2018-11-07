import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
from RL.lib.envs.blackjack import BlackjackEnv
from RL.lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

# Run the episodes

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float)*epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor = 1.0, epsilon = 0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function 
    # A nested dictionary that maps state -> (action -> action-value)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episodes in range(num_episodes):
        if i_episodes % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episodes, num_episodes), end="")
            sys.stdout.flush()
        
        # generate an episode. This will be a list containing the tuple
        # (state, action, reward) for each time step during the episode
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p = probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            # Terminate the episode when done
            if done: 
                break
            # update state for next time step
            state = next_state 

        # Find state-action pairs that occurred in the episode
        sa_pair_visited = set([(tuple(x[0]), x[1]) for x in episode])
        
        for state, action in sa_pair_visited:
            # find the first occurrence of the state action pair in the episode
            first_occurrence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            # sum up all rewards since the first occurrance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurrence_idx:])])
            
            sa_pair = (state, action)
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair]/returns_count[sa_pair]

    return Q, policy
    
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon = 0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")