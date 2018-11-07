import gym
import itertools
import matplotlib
import numpy as np 
import pandas as pd 
import sys

from  collections import defaultdict
from RL.lib.envs.cliff_walking import CliffWalkingEnv
from RL.lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

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

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-learning algorithm: Off-policy TD control. Finds the optimal greedy policy 
    while following an epsilon-greedy policy. 

    Args: 
        env: OpenAI environment
        num_episodes: Number of episodes to run for
        discount_factor: Gamma discount factor
        alpha: TD learning rate
        epsilon: The probability of choosing a random action. Float between 0 and 1

    Returns: 
        A tuple (A, episode_lengths)
        Q is the optimal action-value function, a dictionary mapping state -> action_values. 
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode rewards. 
    """

    # The final action-value function. 
    # A nested dictionary that maps state -> (action -> action-value)
    Q = defaultdict(lambda: np.ones(env.action_space.n))

    # Keeps trac of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))
    
    # The policy we are currently following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out the episode currently running
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode + 1, num_episodes),end="")
            sys.stdout.flush()

        # Initialise S
        state = env.reset()

        for t in itertools.count(): 
            # Render environment on the last episode
            if i_episode == (num_episodes - 10):
                print("Last Episode")
                env.render()

            # Find action probabilities
            action_prob = policy(state)
           
            # epsilon-greedy choice of action
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            
            # Take action A, observe R, S'
            next_state, reward, done, _ = env.step(action)

            # Update episode statistics
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # Update Rule: 
            next_best_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor*Q[next_state][next_best_action]
            td_delta  = td_target - Q[state][action]
            Q[state][action] += alpha*td_delta
            
            if done: 
                break
            
            # Update state for next iteration
            state = next_state
    return Q, stats

Q, stats = q_learning(env, 500)

plotting.plot_episode_stats(stats)