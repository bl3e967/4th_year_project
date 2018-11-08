import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import sklearn.preprocessing
from RL.lib import plotting 
from sklearn.linear_model import SGDRegressor 
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

class q_learning():
    """
        Function which uses function approximation Q-learning. 
        
        Args: 
            env: OpenAI environment. 
            epsilon: epsilon-greedy policy
            epsilon_decay_rate: epsilon decay rate
            discount_factor: gamma discount factor
            num_episodes: number of episodes to run during training
            estimator: Class for estimating value function

        returns: 
            Renders the environment after a criteria has been met
            Training statistics 
    """
    def __init__(self, env, epsilon=0.1, epsilon_decay_rate=0.1, discount_factor=1.0, num_episodes=10000, render = True, render_after = 1000, render_every = 50):
        # AI gym environment
        self.env = env
        # Render environment
        self.render_flag = render
        self.render_after = render_after
        self.render_interval = render_every
        # Hyperparameters
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
    
    class estimator():
        def __init__(self):
            # Need to compute mean and variance of action space to normalize.
            # To do this, take samples from observation space. 
            samples = np.array([np.clip(env.observation_space.sample(), -99999, 99999) for x in range(10000)])
            # Scaler will find the mean and variance of the observation space using the
            # samples that has been generated

            self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler.fit(samples)
            
            # Feature vector composed of Radial Basis Functions of differnet gamma's.
            # Reference pipeline: http://sciksit-learn.org/dev/modules/compose.html
            # Reference RBF: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler
            self.feature_vector = FeatureUnion([
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.7, n_components=100)),
                ('rbf5', RBFSampler(gamma=0.5, n_components=100)),
                ('rbf6', RBFSampler(gamma=0.2, n_components=100)),
                ('rbf7', RBFSampler(gamma=0.1, n_components=100))
            ])

            # Now initialize the feature vector by fitting it to the initial samples from the
            # observation space. The samples are normalised first using the mean and variance
            # computed earlier. 
            self.feature_vector.fit(self.scaler.transform(samples))

            # Have a sgd object for each action
            self.optimizers = []
            for _ in range(env.action_space.n):
                optimizer = SGDRegressor()
                optimizer.partial_fit(self.featurize_state(env.reset()), [0])
                self.optimizers.append(optimizer)
        
        def featurize_state(self, state):
            # Convert a state into features
            scaled = self.scaler.transform([state])
            featurized = self.feature_vector.transform(scaled)
            return featurized 

        def predict_q(self, state, action = None):
            # Featurize state
            feature = self.featurize_state(state)
            # Predict q-value for each action
            if action is None:
                return np.array([m.predict(feature)[0] for m in self.optimizers])
            else: 
                return self.optimizers[action].predict(feature)

        def update(self, state, action, target):
            """
                Update parameters using sgd
            """
            feature = self.featurize_state(state)
            self.optimizers[action].partial_fit(feature, [target])                

    def get_epsilon(self, time):
        return self.epsilon*self.epsilon_decay_rate**time
    
    def make_epsilon_greedy_policy(self, q_value, state, epsilon):
        action_prob = np.ones(self.env.action_space.n)*epsilon/self.env.action_space.n 
        best_actions = np.argmax(q_value)
        action_prob[best_actions] += (1.0 - epsilon)
        return action_prob        
        
    def run(self):
        training_stats = plotting.EpisodeStats(
            episode_lengths = np.zeros(self.num_episodes),
            episode_rewards = np.zeros(self.num_episodes)
        )
        estimator = q_learning.estimator()
        average_length = 200.0
        
        for i_episodes in range(self.num_episodes):
            # Initialise state
            state = self.env.reset()
            for t in itertools.count():           
                # Find action given by current policy for state
                # q_value here is an array of values for each action
                if self.render_flag:
                    if i_episodes > self.render_after and i_episodes % self.render_interval == 0:
                        self.env.render()

                q_value = estimator.predict_q(state, action=None)
                epsilon = self.get_epsilon(i_episodes)
                # policy is an array of probabilities for each action
                policy = self.make_epsilon_greedy_policy(q_value, state, epsilon)
                # Choose action epsilon-greedily
                best_action = np.random.choice(np.arange(len(policy)), p=policy)
                # Take best_action, observe reward and next_state
                next_state, reward, done, _ = self.env.step(best_action)

                training_stats.episode_rewards[i_episodes] += reward

                # Update equation: 
                # Here, estimator predicts a single value for the state, action pair
                td_target = reward + self.discount_factor*np.max(estimator.predict_q(next_state))
                # Estimator updates parameters
                estimator.update(state, best_action, td_target)

                if done: 
                    training_stats.episode_lengths[i_episodes] = t
                    average_length = np.sum(training_stats.episode_lengths)/i_episodes
                    break

                state = next_state

                print("\rEpisode {}, time {}. Last Episode Reward: {}. Average length = {}".format(
                    i_episodes, t, training_stats.episode_rewards[i_episodes - 1], average_length), end="")
                sys.stdout.flush() 

        return training_stats


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    MountainCarSolver = q_learning(env, num_episodes=10000, epsilon=1.0, epsilon_decay_rate=0.1, discount_factor = 0.99, render=True, render_after=1000)
    stats = MountainCarSolver.run()
    plotting.plot_episode_stats(stats, smoothing_window = 25)

    # Train for a few hours, see if it converges. If it converges, then maybe try to make it faster. Else, then there is a problem that
    # should be addressed. 