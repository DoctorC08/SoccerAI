import gym
from gym.spaces import Box, Tuple, Discrete
from Enviornment import env
import numpy as np
Env = env()

state = Env.reset()
n_observations = len(state)
print(state)
print(n_observations)


print(Env.action_space)
print((4 * 3 * 2 * 3) ** 2) # Action space possibilities

