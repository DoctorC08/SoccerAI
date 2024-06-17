# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np

from Enviornment import GridSoccer
from Networks import Agent
from Graphs import Graph

global verbose
verbose = False


array1 = np.array([1, 2])
array2 = np.array([3, 4])
array3 = np.array([5, 6])

print(np.append(array1, np.append(array2, array3)))
print(np.stack([array1, array2, array3]).reshape(1, -1).squeeze())
print(np.concatenate([array1, array2, array3]))
