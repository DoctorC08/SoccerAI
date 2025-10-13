# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np
import random as rand

# from Enviornment import GridSoccer
# from Networks import Agent
# from Graphs import Graph


import gymnasium as gym

env = gym.make("ALE/Tetris-v5", obs_type="grayscale", render_mode="human")
observation = env.reset()

if 'render_fps' not in env.metadata:
    env.metadata['render_fps'] = 60

done = False
while not done:
    env.render()
    
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    

env.close()