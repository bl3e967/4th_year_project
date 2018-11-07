import numpy as np
import value_iteration as vi
import gym 
from gym import wrappers as wrapper

simulation_length = 100
environment = 'FrozenLake8x8-v0'
directory   = 'VI-frozenlake8x8-1'
env = gym.make(environment)
env = wrapper.Monitor(env, directory, force = True, video_callable=lambda episode_id: True)
# Find optimal value function and the optimal policy using value iteration
gamma = 1.0 
optimalPol, optimalV = vi.value_iteration(env, discount_factor = gamma)

# Initialise container for reward record
reward_for_av = []

for episode in range(simulation_length):
    observation = env.reset()
    for t in range(10000):  
        env.render() 
        action = optimalPol[observation]
        observation, reward, done, info = env.step(action)
        if done: 
            if reward == 0.0:
                print("LOSE")
            else:
                print("WIN")
            print("Episode finished after {} timesteps".format(t+1))
            break
        reward_for_av.append(reward)
        # print average reward every 1000 steps
        if simulation_length % 1000 == 0:
            print('Current average reward: %f' % np.mean(reward_for_av))
env.close()