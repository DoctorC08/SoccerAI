import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

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
lists = [5, 6, 7]
for i, l in enumerate(lists):
    print(i, l)
