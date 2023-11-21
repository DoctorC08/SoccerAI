# main.py
# by christophermao
# 11/15/23

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

# Allow updates to graph as training progresses
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



from Networks import Agent
from Enviornment import env
Env = env()
obs = Env.reset()

agent = Agent(Env)

agent.train(num_episodes=5)
