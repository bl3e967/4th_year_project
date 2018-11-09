import gym
import numpy as np 
import itertools
import tensorflow as tf

class dq_learner():
    def __init__(self, env, memory_capacity = 1e6, init_memory_capacity = 50000):
        self.env = env 
        self.memory_capacity = memory_capacity
        self.init_memory_capacity = init_memory_capacity
    
    def _initialise_memory():
        return None 

    def q_learning(self, epsilon, discount_factor = 0.99, num_episodes = 10000):
        # Initialise Replay memory D to capacity N
        self._initialise_memory()

        # TODO: initialise action-value function Q with random weights theta 

        # TODO: Initialise target action-value function Q' with weights theta' = theta

        for i_episode in range(num_episodes):
            # Initialise 
