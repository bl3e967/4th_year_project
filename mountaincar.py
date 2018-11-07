import gym
import numpy as np 
import sys

env = gym.make('MountainCar-v0')

for t in range(1000):
    env.render()