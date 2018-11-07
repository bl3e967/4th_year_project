# implementation of SARSA algorithm for windy gridworld environment

import gym
import itertools
import matplotlib
import numpy as np 
import pandas as pd 
import sys

from collections import defaultdict
from RL.lib.envs.windy_gridworld import WindyGridworldEnv
from RL.lib import plotting

matplotlib.style.use('ggplot') 

env = WindyGridworldEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon greedy policy based on a given Q function and epsilon

    Args:
        Q: A dictionary that maps from state -> action-values
           Each value is a numpy array of length nA
        epsilon: The probability to select a random action. float between 0 and 1
        nA: Number of actions in the environment

    Returns: 
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float)*epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA Algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy

    Args:
        env: OpenAI environment
        num_episodes: Number of episodes to run for
        discount_factor: Gamma discount factor
        alpha: TD learning rate
        epsilon: The probability of taking a random action. Float between 0 and 1

    Returns: 
        A: tuple (Q, stats)
        Q: the optimal actoin-vallue function, a dictionary mapping state->action values. 
        stats in an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards. 
    """
    # The final action-value function. 
    # A nested dictionary that maps state -> (action -> action-value). 
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # print out episode number
        
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        for i in itertools.count():
            # Render environment
            if i_episode == (num_episodes - 1):
                print("Last Episode")
                env.render()

            # Take a step
            next_state, reward, done, _ = env.step(action)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
            
            # Update Q value estimate
            td_target = reward + discount_factor*Q[next_state][next_action] 
            td_delta  = td_target - Q[state][action]
            Q[state][action] +=  alpha*td_delta
            
            # update statistics
            stats.episode_lengths[i_episode] = i
            stats.episode_rewards[i_episode] += reward

            if done: 
                break

            # Update state and action for next step
            state  = next_state
            action = next_action        
    return Q, stats

Q, stats = sarsa(env, 200)
plotting.plot_episode_stats(stats)


        