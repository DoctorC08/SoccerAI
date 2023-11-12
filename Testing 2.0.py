import gym
from gym.spaces import Tuple, Discrete, Box
import numpy as np

list = [1, 2, 4, 5]
list2 = [12345456, 56, 3, 3]
list = [list[i] + list2[i] for i in range(len(list2))]
print(list)
