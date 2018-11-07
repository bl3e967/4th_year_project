import gym
import numpy as np
import sys
from matplotlib import pyplot as plt

class deep_q_learner():
    def __init__(self, 
                env,
                num_episodes = 10000, 
                epsilon = 1.0, 
                epsilon_decay_rate = 1e-6,
                epsilon_threshold = 0.1, 
                discount_factor = 0.99, 
                replay_memory_max_capacity = 500000, 
                replay_memory_initial = 50000,
                frame_skipping = 4):
        
        # OpenAI environment
        self.env = env 
        
        # Training parameters
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_threshold = epsilon_threshold
        self.discount_factor = discount_factor
        self.replay_memory_max_capacity = sreplay_memory_max_capacity
        self.replay_memory_initial = replay_memory_initial
        self.frame_skipping = frame_skipping

    def preprocesser(self, image, m = 4):
        # 1. take the max value for each pixel colour value over the 
        # frame being encoded and the previous frame

        # 2. Extract the Y channel (luminance) from RGB frame and rescale
        # to 84 x 84. Apply to the m most recent frames and stack them
        # to produce the input to the Q-function
        return None

    def q_learning(self):
        
    