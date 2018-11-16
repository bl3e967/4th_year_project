import gym
import numpy as np 
import itertools
import tensorflow as tf

class dq_learner():
    def __init__(self, env, summary_dir = None):
        self.env = env 
        self.estimator = self.estimator(self.env)
        self.summary_dir = summary_dir 
    
    class estimator():
        def __init__(self, weights = None, env):
            self.env = env 
            # If weights = None, then build graph
            if weights is None:
                # Initialise graph
                with tf.variable_scope("Estimator"):
                    self._build_graph()

                    # Create filewriter object if output directory has been specified
                    if self.summary_dir: 
                        output_dir = os.path.join(summary_dir, "summaries_{}".format("Estimator"))
                        if not os.path.exists(summary_dir):
                            os.makedirs(summary_dir)
                        self.summary_writer = tf.summary.FileWriter(summary_dir)

            
            # If weights != None, then use these values as the new weights
            else:
                # Do something
        
        def _build_graph(self):
            """
            Build Convolutional Neural Network:
            
                The input to the neural network consists of an 84, 84, 4 image 
                produced by the preprocessing map w. 
                
                The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image 
                and applies a rectifier nonlinearity. 
                
                The second hidden layer convolves 64 filters of 4 x 4 with stride 2, 
                again followed by a rectifier nonlinearity.
                
                This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with
                stride 1 followed by a rectifier. 
                
                The final hidden layer is fully-connected and consists of 512 rectifier units. 
                
                The output layer is a fully-connected linear layer with a single output for each valid action.
            """
            # Placeholder for input [batch_size, 84, 84, m = 4]
            self.input_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype = uint8, name = "Input_State")

            # Placeholder for output [Depends on valid actions]
            self.target_pl = tf.placeholder(shape=[None], dtype=tf.float32, name = "td_target")

            # Placeholder for actions that were selected
            self.actions_pl = tf.palceholder(shape=[None], dtype = tf.int32, name-"actions")

            # Normalise input image
            input_image = tf.to_float(self.input_pl)/255.0
            
            # Get batch size
            batch_size = tf.shape(tf.input_pl)[0]

            # Convolutional layers
            layer1 = tf.contrib.layers.conv2d(input_image, 32, 8, 4, activation_fn=tf.nn.relu)
            layer2 = tf.contrib.layers.conv2d(layer1, 64, 4, 2, activation_fn=tf.nn.relu)
            layer3 = tf.contrib.layers.conv2d(layer2, 64, 3, 1, activation_fn=tf.nn.relu)        
            
            # Input to fully connected layer has to be dimension [batch size, depth]
            # Currently our ouput is [batch size, width, height, depth] and so we need to flatten
            fcinput = tf.contrib.layers.flatten(layer3)           
            fclayer = tf.contrib.layers.fully_connected(fcinput, 512)
            self.predictions = tf.contrib.layers.fully_connected(fclayer, len(self.env.action_space.n))

            # Get predictions for chosen actions only
            indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.action_pl
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), indices)

            # Find Loss values. Optimizer specified in paper as RMSPropOptimizer
            self.loss_values = tf.squared_difference(self.target_pl, self.action_predictions)
            
            # Find mean of loss values
            self.loss = tf.reduce_mean(self.loss_values)

            # Optimise using RMS prop. Optimiser from paper
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            # Increment global step by one once minimized
            self.train_op  = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
            
            # Compile Tensorboard summaries
            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.predictions),
            ])

        def preprocess(self, sess, image):
            with tf.variable_scope("PreProcessor"): 

            # Placeholder for input image
            self.image = tf.placeholder(shape=[210,160,3], dtype=uint8)
            
            # Convert to grayscale to reduce computation as colour is not important
            self.image = tf.image.rgb_to_grayscale(self.image)
            
            # Crop image to get rid of irrelevant image pixels
            self.image = tf.image.crop_to_bounding_box(self.image, 34, 0, 160, 160)
            
            # Resize image to [84,84] using nearest-neighbour method
            self.state = tf.image.resize_images(self.image, [84,84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
            return sess.run(self.state, { self.image = image })

        def predict(self, sess, state):
            """
                Predict the q_values for each possible action in the current state
                
                Args:
                    state: The current state

                Return: 
                    An array containing the Q values for the corresponding actions
            """
            return sess.run(self.predictions, { self.input_pl: state })

        def update(self, sess, state, action, td_target):
            feed_dict = {
                self.input_pl  = state,
                self.action_pl = action,
                self.target_pl = td_target
            }
            summaries, loss, _ , global_step = sess.run(
                [self.summaries, self.loss, self.train_op, tf.contrib.framework.get_global_step()],
                feed_dict
                )
            
            if self.summary_writer:
                self.summary_writer.add_summary(summaries, global_step)

            return loss
            
        def make_epsilon_greedy_policy(self, state):

    class memory_container():
        def __init__(self, sess, env, init_memory_capacity, batch_size, q_estimator):
            self.env = env
            self.D = []
            state = q_estimator.preprocess(self.env.reset())
            
            # Initialise state by concatenating m = 4 initial frames together 
            state = np.stack([state]*4, axis=2)

            # Initialise memory to capacity N
            for episode_no in range(init_memory_capacity):
                action_probs = q_estimator.make_epsilon_greedy_policy(state)
                best_action  = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                next_frame, reward, done, _ = self.env.step(best_action)
                next_state = q_estimator.preprocess(next_frame)

                # Remove oldest frame and append new frame. expand_dims needed to concatenate
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                self.D.append(tuple(state, next_state, action, reward))

                if done: 
                    state = q_estimator.preprocess(env.reset())
                    state = np.stack([state]*4, axis = 2)
                else: 
                    state = next_state
    
        def append(self, transition):
            self.D.append(transition)
            return None 

        def sample(self):
            # sample minibatch of size batch_size
            minibatch = random.sample(self.D, self.batch_size)
            return minibatch

    def q_learning(self, sess, epsilon, discount_factor = 0.99, num_episodes = 10000, memory_capacity = 1e6, init_memory_capacity = 50000):
        # Initialise action-value function Q with random weights theta
        q_estimator = estimator()
        
        # Initialise Replay memory D to capacity N
        # D is a list of tuples (state, next_state, action, reward)
        D = memory_container(sess, env, init_memory_capacity, batch_size, q_estimator)

        # Initialise target action-value function Q' with weights theta' = theta
        target_estimator = estimator(q_estimator.get_weights())

        for i_episode in range(num_episodes):
            # Initialise state
            state = q_estimator._preprocess(sess, self.env.reset())

            # With probability epsilon select a random action
            # Otherwise select action = argmax(Q(image,action)|theta)
            action_prob = q_estimator.make_epsilon_greedy_policy(state)
            best_action = np.random.choice(np.arange(action_prob), p=action_prob)

            # Execute action in emulator and observe reward and image
            next_image, reward, done, _ = self.env.step(best_action)
            
            # Preprocess next_state and set next-state = state, action, reward
            next_state = self._preprocess(next_image)

            # Make a tuple of (state, next_state, action, reward) and store it in D
            D.append(tuple(state, next_state, action, reward))

            # Sample random minibatch of transitions from D
            minibatch = D.sample()

            # map(myfunc, iterable): executes myfunc for each item in iterable
            # TODO: zip(*iterable):  
            states, actions, rewards, next_states = map(np.array, zip(*minibatch))

            # Set td_target = reward_j if episode terminates at j+1
            # set td_target = reward_j + gamma*max(Q'(next_state, action'|theta'))
            if done: 
                td_target = reward
            else:
                q_values  = target_estimator.predict(sess, next_state) 
                td_target = reward + discount_factor*np.max(q_values)
            
            # Perform a gradient descent step on td_error wrt theta
            loss = target_estimator.update(sess, state, action, td_target)

            # Every c steps reset Q' = Q



