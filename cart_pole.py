"""
source: https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578
"""
import gym
import numpy as np 
import math
import sys
from collections import deque

class QCartPoleSolver():
    def __init__(self, buckets=(1,1,6,12,), 
        n_episodes = 10000, n_win_ticks = 200, min_alpha = 0.1, 
        min_epsilon=0.1, gamma=1.0, ada_divisor=25, 
        max_env_steps=None, quiet=False, monitor=False ):
        # Down-scale feature space to discrete range
        self.buckets = buckets 
        # Training episodes
        self.n_episodes = n_episodes
        # Average ticks over 100 episodes required for win
        self.n_win_ticks = n_win_ticks
        # learning rate
        self.min_alpha = min_alpha
        # exploration rate
        self.min_epsilon = min_epsilon
        # discount factor
        self.gamma = gamma
        # Only for development purposes
        self.ada_divisor = ada_divisor
        self.render_flag = False 
        self.quiet = quiet

        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None: self.env.max_episode_steps = max_env_steps
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios  = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)
    
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        # Decay epsilon over episode number. Policy becomes deterministic over time
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))
    
    def get_alpha(self, t):
        # Decay learning rate over episode number. 
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
    
    def render_episode(self):
        self.render_flag = True

    def run(self):
        scores = deque(maxlen = 100)

        for i_episodes in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())
            alpha = self.get_alpha(i_episodes)
            epsilon = self.get_epsilon(i_episodes)
            done = False
            i = 0 

            while not done: 
                if self.render_flag is True:
                    self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                next_state = self.discretize(obs)
                self.update_q(current_state, action, reward, next_state, alpha)
                current_state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)

            if mean_score >= 150 and i_episodes >= 100:
                self.render_episode()

            if mean_score >= self.n_win_ticks and i_episodes >= 100:
                if not self.quiet: 
                    print('\rRan {} episodes. Solved after {} trials'.format(i_episodes, i_episodes - 100))
                    sys.stdout.flush()
                    
                return i_episodes - 100
            if i_episodes % 100 == 0 and not self.quiet:
                print('\r[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(i_episodes, mean_score))
                sys.stdout.flush()

        if not self.quiet: print('Did not solve after {} episodes'.format(i_episodes))
        return i_episodes
    
if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()