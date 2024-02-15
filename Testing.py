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

list_data = torch.tensor([1, 2, 3])
newer_list = list_data.clone().detach().requires_grad(True)
print(newer_list.shape)

