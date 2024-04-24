# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm
import wandb

from Enviornment import GridSoccer
from Networks import Agent
from Graphs import Graph

global verbose
verbose = False
