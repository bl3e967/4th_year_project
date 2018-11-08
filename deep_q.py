import gym
import numpy as np
import sys
import tensorflow as tf
import itertools
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
                frame_skipping = 4,
                batch_size = 32):
        
        # OpenAI environment
        self.env = env 
        
        # Training parameters
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_threshold = epsilon_threshold
        self.discount_factor = discount_factor
        self.replay_memory_max_capacity = replay_memory_max_capacity
        self.replay_memory_initial = replay_memory_initial
        self.frame_skipping = frame_skipping

    class estimator(deep_q_learner):
        """
            Q-value estimator using Convolutional Neural Network. 
        """
        def __init__(self):
            with tf.variable_scope("Estimator"):
                # Build tensorflow graph
                self._build_model()
        
        def _build_model(self):
            """ 
                Build tensorflow graph for neural network
                Input: 84 x 84 x m (m = 4) image
                First Hidden Layer: 
                    Convolutional layer: 32 filters (8x8), stride 4
                    ReLu Activation Function
                Second Hidden Layer: 
                    Convolutional Layer: 64 filters (4x4), stride 2
                    ReLu Activation Function 
                Third Hidden Layer: 
                    Convolutional Layer: 64 filters (3x3), stride 1
                    ReLu Activation Function
                Final Hidden Layer: 
                    512 ReLu Activation Functions
                Output Layer: 
                    Fully Connected Linear Layer - single output for each valid action            
            """
            # Placeholder for input. m corresponds to the number of frames being concatenated
            m = 4
            self.image_pl = tf.placeholder(shape=[None, 84, 84, m], dtype=tf.uint8, name="Input")
            # Placeholder for td_target
            self.td_target = tf.placeholder(shape = [None], dtype = tf.float32, name="td_error")
            # Action id
            self.actions_pl = tf.placeholder(shape = [None], dtype=tf.int32, name="actions")

            # Normalize pixel values
            input_image = tf.to_float(self.image_pl)/255
            batch_size = tf.shape(self.image_pl)[0]            

            # Convolutional Layers
            conv1 = tf.contrib.layers.conv2d(input_image, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            # Fully connected layer input
            fcn_input = tf.contrib.layers.flatten(conv3)
            # First fully connected layer
            fc1 = tf.contrib.layers.fully_connected(fcn_input, 512)
            # Output layer
            self.predictions = tf.contrib.layers.fully_connected(fc1, len(env.action_space.n))

            # Get predictions for chosen actions only
            gather_indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.actions_pl

        
        def predict_q(self):
            return None 







    def _preprocessor(self, sess, image):
        """
        Preprocess the images from the emulator: 

            1. take the max value for each pixel colour value over the 
            frame being encoded and the previous frame. Necessary to remove
            flickering in games where some objects appear only in even frames 
            while other objects appear only in odd frames

            2. Extract the Y channel (luminance) from RGB frame and rescale
            to 84 x 84. 
        """
        # Build tensorflow graph for preprocessor
        with tf.variable_scope("Preprocessor"):
            # Convert RGB image to Grayscale
            output = tf.image.rgb_to_grayscale(image)
            # Crop the image to exclude the results section
            output = tf.image.crop_to_bounding_box(output, 40, 0, 160, 160)
            # Resize image to 84 x 84
            output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return sess.run(output)

    def _make_epsilon_greedy_policy(self, state, q_value, time):
        
        def get_epsilon(self, time):
            epsilon = self.epsilon*self.epsilon_decay_rate**time
            if epsilon > self.epsilon_threshold:
                return epsilon
            else:
                return self.epsilon_threshold
        


    def q_learning(self):
        for i_episodes in range(self.num_episodes):
            # Initialise state S, and preprocess
            state = self._preprocessor(self.env.reset())

            for t in itertools.count():
                # Make estimation of Q
                estimator = deep_q_learner.estimator()
                q_value = estimator.predict_q(state)
                
                # Choose action A from S using policy derived from Q epsilon-greedily
                action_probs = self._make_epsilon_greedy_policy(state, q_value, i_episodes)
                best_action  = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                
                # Take action A, observe R, next_state S
                next_state, reward, done, _ = self.env.step(best_action)
                next_state = self._preprocessor(next_state)

                # Update parameter estimation
                
                if done: 
                    break

                state = next_state

