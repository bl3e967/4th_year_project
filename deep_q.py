import gym
import sys
import numpy as np 
import itertools
import os
import random
import tensorflow as tf

from gym.wrappers import Monitor

class dq_learner():
    class estimator():
        def __init__(self, env, variable_scope_name, summary_dir = None):
            self.env = env 
            self.name = variable_scope_name
            
            with tf.variable_scope(variable_scope_name):
                # Construct graph
                self._build_graph()

                # Create filewriter object if output directory has been specified
                if summary_dir is not None: 
                    summary_dir = os.path.join(summary_dir, "summaries_{}".format(variable_scope_name))
                    
                    if not os.path.exists(summary_dir):
                        os.makedirs(summary_dir)
                    
                    self.summary_writer = tf.summary.FileWriter(summary_dir)
                else:
                    self.summary_writer = None 
        
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
            self.input_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name = "Input_State")
            # Placeholder for output [Depends on valid actions]
            self.target_pl = tf.placeholder(shape=[None], dtype=tf.float32, name = "td_target")
            # Placeholder for actions that were selected
            self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            # Normalise input image
            input_image = tf.to_float(self.input_pl)/255.0
            
            # Get batch size
            batch_size = tf.shape(self.input_pl)[0]

            # Convolutional layers
            layer1 = tf.contrib.layers.conv2d(input_image, 32, 8, 4, activation_fn=tf.nn.relu)
            layer2 = tf.contrib.layers.conv2d(layer1, 64, 4, 2, activation_fn=tf.nn.relu)
            layer3 = tf.contrib.layers.conv2d(layer2, 64, 3, 1, activation_fn=tf.nn.relu)        
            
            # Input to fully connected layer has to be dimension [batch size, depth]
            # Currently our ouput is [batch size, width, height, depth] and so we need to flatten
            fcinput = tf.contrib.layers.flatten(layer3)           
            fclayer = tf.contrib.layers.fully_connected(fcinput, 512)
            self.predictions = tf.contrib.layers.fully_connected(fclayer, self.env.action_space.n)

            # Get predictions for chosen actions only
            indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.actions_pl
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), indices)

            # Find Loss values. Optimizer specified in paper as RMSPropOptimizer
            self.loss_values = tf.squared_difference(self.target_pl, self.action_predictions)
            
            # Find mean of loss values
            self.loss = tf.reduce_mean(self.loss_values)

            # Optimise using RMS prop. Optimiser from paper
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            # Increment global step by one once minimized
            self.train_op  = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            
            # Compile Tensorboard summaries
            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
                tf.summary.histogram("loss_hist", self.loss),
                tf.summary.histogram("q_values_hist", self.predictions),
            ])

        # def get_weights(self, sess):
        #     # Create array of trainable parameters from original network
        #     param = [theta for theta in tf.trainable_variables() if theta.name.startswith("Estimator")]
        #      # Sort parameters
        #     param = sorted(param, key=lambda v: v.name)
        #     return sess.run(param)

        def make_epsilon_greedy_policy(self, sess, state, epsilon):
            action_probs = np.ones(self.env.action_space.n, dtype=float)*epsilon/self.env.action_space.n
            q_value = self.predict(sess, np.expand_dims(state, 0))[0]
            best_actions_indices = np.argmax(q_value)
            action_probs[best_actions_indices] += (1.0 - epsilon)
            return action_probs

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
                self.input_pl  : state,
                self.actions_pl : action,
                self.target_pl : td_target
            }
            summaries, loss, _ , global_step = sess.run(
                [self.summaries, self.loss, self.train_op, tf.train.get_global_step()],
                feed_dict)
            
            if self.summary_writer is not None:
                self.summary_writer.add_summary(summaries, global_step)

            return loss
    
    class ModelParametersCopier():
        """
        Copy model parameters of one estimator to another
        From: https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning%20Solution.ipynb
        """
        def __init__(self, estimator1, estimator2):
            """
            Define copy-work operation graph.
            Args: 
                estimator1: Estimator to copy the parameters from
                estimator2: Estimator to copy the parameters to
            """
            e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.name)]
            e1_params = sorted(e1_params, key=lambda v: v.name)
            e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.name)]
            e2_params = sorted(e2_params, key=lambda v:v.name)

            self.update_ops = []
            for e1_v, e2_v in zip(e1_params, e2_params):
                op = e2_v.assign(e1_v)
                self.update_ops.append(op)

        def make(self, sess):
            """
            Makes a copy. 
            Args: 
                sess: Tensorflow session instance
            """
            sess.run(self.update_ops)
    
    class PreProcessor():
        def __init__(self):
            with tf.variable_scope("PreProcessor"): 

                # Placeholder for input image
                self.input = tf.placeholder(shape=[210,160,3], dtype=tf.uint8)
                
                # Convert to grayscale to reduce computation as colour is not important
                self.image = tf.image.rgb_to_grayscale(self.input)
                
                # Crop image to get rid of irrelevant image pixels
                self.image = tf.image.crop_to_bounding_box(self.image, 34, 0, 160, 160)
                
                # Resize image to [84,84] using nearest-neighbour method
                self.state = tf.image.resize_images(self.image, [84,84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                self.state = tf.squeeze(self.state)
            
        def preprocess(self, sess, image):
            return sess.run(self.state, { self.input : image })
    
    class Memory_Container():
        def __init__(self):
            self.D = []
        
        def populate_memory(self, sess, env, PreProcessor, init_memory_capacity, q_estimator, batch_size, epsilon):
            self.env = env
            self.batch_size = batch_size
            state = self.env.reset()
            state = PreProcessor.preprocess(sess, state)
            
            # Initialise state by concatenating m = 4 initial frames together 
            state = np.stack([state]*4, axis=2)

            # Initialise memory to capacity N
            for i in range(init_memory_capacity):
                print("\rInitialising memory capacity {}/{}".format(i,init_memory_capacity), end="")
                sys.stdout.flush()
                action_probs = q_estimator.make_epsilon_greedy_policy(sess, state, epsilon[i])
                best_action  = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                next_frame, reward, done, _ = self.env.step(best_action)
                next_state = PreProcessor.preprocess(sess, next_frame)

                # Remove oldest frame and append new frame. expand_dims needed to concatenate
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                self.D.append(tuple([state, next_state, best_action, reward, done]))

                if done: 
                    # TODO: Doing env.reset inside preprocess seems to do something weird
                    state = env.reset()
                    state = PreProcessor.preprocess(sess, state)
                    state = np.stack([state]*4, axis = 2)
                else: 
                    state = next_state
            
            print("\nDone...")
    
        def append(self, transition):
            self.D.append(transition)
            return None 
        
        def pop(self, index):
            self.D.pop(index)
            return None

        def get_length(self):
            return len(self.D)

        def sample(self):
            # sample minibatch of size batch_size
            minibatch = random.sample(self.D, self.batch_size)
            return minibatch

    def __init__(self, 
                 env, 
                 summary_dir = None,
                 checkpoint_dir = None,
                 video_dir = None, 
                 record_video_every = 50, 
                 epsilon = 1.0,
                 epsilon_decay_over=500000, 
                 min_epsilon = 0.1,
                 discount_factor = 0.99, 
                 num_episodes = 10000, 
                 memory_capacity = 500000, 
                 init_memory_capacity = 50000, 
                 batch_size = 32, 
                 update_network_every = 10000):
        # OpenAI gym environment
        self.env = env 
        
        # # Directory to output summaries
        # self.summary_dir = summary_dir
        # # Directory to pre-exting checkpoint directory
        # self.checkpoint_dir = checkpoint_dir
        # # Directory to output video recording
        # self.video_dir = video_dir

        self._check_directories(summary = summary_dir,
                                checkpoint = checkpoint_dir,
                                video = video_dir)
        
        
        self.video_record_interval = record_video_every
        # Epsilon and epsilon decay rate and minimum epsilon value
        # Changed epsilon profile as np.linspace rather than decay over
        # episode number
        self.epsilon_decay_step_number = epsilon_decay_over
        self.min_epsilon = min_epsilon
        self.epsilon = np.linspace(epsilon, min_epsilon, epsilon_decay_over)
        # Discount factor
        self.discount_factor = discount_factor
        # Total number of episodes to train on
        self.num_episodes = num_episodes
        # Total capacity for experience memory
        self.memory_capacity = memory_capacity
        # Experience memory capacity for initialisation
        self.init_memory_capacity = init_memory_capacity
        # Batch size for experience batch updates
        self.batch_size = batch_size
        # The interval between updates for td_target estimator network
        self.network_update_interval = update_network_every
        # Initialise loss value
        self.loss = None

        # Initialise Q value estimator objects
        self.q_estimator      = self.estimator(self.env, "q_estimator",  summary_dir = self.summary_dir)
        self.target_estimator = self.estimator(self.env, "td_estimator", summary_dir = self.summary_dir)
        
        # Initialise memory container objects
        self.D = self.Memory_Container()

        # Initialise image preprocessing object
        self.preprocessor = self.PreProcessor()

        # Initialise model parameters copier
        self.copier = self.ModelParametersCopier(self.q_estimator, self.target_estimator)

    def _check_directories(self, summary, checkpoint, video = None):
        if summary and checkpoint and video is None:
            raise IOError('At least summary and checkpoint directory needs to be specified') 
        
        self.checkpoint_dir = os.path.join(checkpoint)
        self.summary_dir    = os.path.join(summary)

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        
        if video is not None:
            self.video_dir = os.path.join(video)
            if not os.path.exists(summary_dir):
                os.mkdir(summary_dir)
        
    # def _get_epsilon(self, episode_no):
    #     """
    #     Calculate value for epsilon. Clip epsilon value to minimum 0.1
    #     Args: 
    #         episode_no: The number of episodes performed so far

    #     Returns: 
    #         epsilon: epsilon value
    #     """
    #     decayed_epsilon = self.epsilon*self.epsilon_decay_rate**episode_no
        
    #     if decayed_epsilon < self.min_epsilon:
    #         return self.min_epsilon
    #     else: 
    #         return decayed_epsilon

    def train(self, sess):               
        if self.init_memory_capacity > self.epsilon_decay_step_number:
            raise ValueError(
                'Initial memory capacity should be equal to epsilon decay step number')
        # Prepopulate bank of memory for experience replay
        self.D.populate_memory(sess, 
                               self.env, 
                               self.preprocessor, 
                               self.init_memory_capacity,
                               self.q_estimator, 
                               self.batch_size,
                               self.epsilon)

        # Initialise tensorflow saver object
        saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        print("\nChecking {} for previous checkpoints".format(self.checkpoint_dir))
        
        if checkpoint: 
            print("\nLoading model checkpoint from {}".format(checkpoint))
            saver.restore(sess, checkpoint)        
        else: 
            print("\nNo previous checkpoint found...")

        # Get global step
        cum_time = sess.run(tf.contrib.framework.get_global_step())

        # Initialise container for episode statistics
        episode_lengths = []
        episode_rewards = []

        if self.video_dir is not None:
            # Record videos
            print("\nCreating Monitor for recording episodes...")
            self.env = Monitor(self.env, 
            directory=self.video_dir, 
            video_callable=lambda count: count % self.video_record_interval == 0, 
            resume=True)

        print("\n Starting Episodes")
        for i_episodes in range(self.num_episodes):
            # Save current state
            saver.save(tf.get_default_session(), self.checkpoint_dir)
            
            # Append new slot for reward list
            episode_rewards.append(0)

            # Initialise state
            state = self.env.reset()
            state = self.preprocessor.preprocess(sess, state)

            # Initialise stack of frames by duplicating the same frame 4 times
            state = np.stack([state]*4, axis = 2)

            for t in itertools.count():
                # Update target estimator every n steps              
                if cum_time % self.network_update_interval == 0:
                    self.copier.make(sess)
                    print("\nTarget Estimator has been updated")

                # With probability epsilon select a random action
                # Otherwise select action = argmax(Q(image,action)|theta)
                # Calculate epsilon value
                if cum_time < self.epsilon_decay_step_number:
                    epsilon = self.epsilon[cum_time]
                else:
                    epsilon = self.min_epsilon

                action_prob = self.q_estimator.make_epsilon_greedy_policy(sess, state, epsilon)
                best_action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                print("\nBest Action {}".format(best_action))
                # Print out episode and iteration
                print("\rEpisode: {}, iteration: {}, Epsilon: {}, loss: {} ".format(
                    i_episodes, t, epsilon, self.loss),
                    end="")
                sys.stdout.flush()

                # Execute action in emulator and observe reward and image
                next_image, reward, done, _ = self.env.step(best_action)
                
                # Update episode statistics container
                episode_rewards[i_episodes] += reward

                # Preprocess next_state and set next-state = state, action, reward
                next_state = self.preprocessor.preprocess(sess, next_image)

                # Append the new state to the end of the stack of states while removing 
                # the oldest state (state[:,:,0])
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis = 2)

                # Discard oldest memory if the memory capacity is full
                if self.D.get_length() == self.memory_capacity:
                    self.D.pop(0)    
                
                # Make a tuple of (state, next_state, action, reward) and store it in D
                self.D.append(tuple([state, next_state, best_action, reward, done]))

                # Sample random minibatch of transitions from D
                minibatch = self.D.sample()

                # map(myfunc, iterable): executes myfunc for each item in iterables  
                states, next_states, actions, rewards, dones = map(np.array, zip(*minibatch))

                # Set td_target = reward_j if episode terminates at j+1
                # set td_target = reward_j + gamma*max(Q'(next_state, action'|theta'))
                q_values   = self.target_estimator.predict(sess, next_states) 
                td_targets = rewards + np.invert(dones).astype(np.float32)*self.discount_factor*np.amax(q_values, axis = 1)
                
                # Perform a gradient descent step on td_error wrt theta
                states = np.array(states)
                self.loss = self.q_estimator.update(sess, states, actions, td_targets)

                # Break if episode is finished
                if done: 
                    # update episode length container
                    episode_lengths.append(t)
                    print("\nEpisode reward : {}".format(episode_rewards[i_episodes]))
                    break

                # Update new state
                state = next_state
                
                # Increment total training time
                cum_time += 1

            # Initialise summaries to load to tensorboard
            episode_summary = tf.Summary()
            # Epsilon summary
            episode_summary.value.add(simple_value = epsilon, tag="episode/epsilon")
            # Reward summary
            episode_summary.value.add(simple_value = episode_rewards[i_episodes-1], tag = "episode/reward")
            # Episode lengths
            episode_summary.value.add(simple_value = episode_lengths[i_episodes-1], tag = "episode/lengths")
            
            if self.q_estimator.summary_writer: 
                self.q_estimator.summary_writer.add_summary(episode_summary, i_episodes)
                self.q_estimator.summary_writer.flush()

if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    
    summary_dir = ".\\Breakout\\summaries\\"
    checkpoint_dir = ".\\Breakout\\checkpoints\\"
    video_dir = ".\\Breakout\\recordings\\"

    tf.reset_default_graph()

    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    VALID_ACTIONS = [0, 1, 2, 3]

    learner = dq_learner(env,
                        summary_dir = summary_dir,
                        checkpoint_dir = checkpoint_dir,
                        video_dir = video_dir,
                        init_memory_capacity=500)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        learner.train(sess)
