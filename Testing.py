import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, combinations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import Agent
from Enviornment import env

import copy

import pandas as pd
from tqdm import tqdm
import time

import time

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import copy

import time
from tqdm.auto import tqdm

list_data = [[1, 2, 3, 4, 20], [11, 12, 13, 14, 15]]


def moving_average(a,
                   n=3):  # Credit: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_mov_avg_lines(data, labels, filter=100):
    for i in range(len(data)):
        new_data = moving_average(data[i], n=filter)
        plt.plot(new_data)

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.show()


# plot_mov_avg_lines(list_data, ["test", "test", "test"], filter=2)
data = torch.tensor([i for i in range(10)])
print(data)
