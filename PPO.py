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

class PPO(nn.module):
    def __init__(self, n_obs, n_acts, fc_layer=32):
        super(PPO, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(n_observations, affine=False, track_running_stats=True),
            nn.Linear(n_observations, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, n_actions),
        )
        
