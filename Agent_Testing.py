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

import numpy as np

# from Enviornment import env
import torch

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

import pandas as pd
import time
global verbose
verbose = False



def map_output_to_actions(output):
    if verbose:
        print("output:", output)
        print("pre action values:", output)
        print("output: ", output)
    return_values = []
    floor_value = output // 2
    if output % 2 == 0:
        return_values.append([floor_value, 0])
    else:
        return_values.append([floor_value, 1])
    if verbose:
        print("return action values:", torch.tensor(return_values))
    return torch.tensor(return_values)

def train(Agents, env, render_mode, num_episodes=50):
    total_rewardsa = []
    total_rewardsb = []
    for i_episode in range(num_episodes):
        observation, _ = env.reset()
        observation = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)

        for t in count():
            observation = observation.squeeze(0)
            with torch.no_grad():
                action1 = torch.argmax(Agents[0](observation))
                action2 = torch.argmax(Agents[1](observation))

            # Convert actions from numbers into lists
            new_action1 = map_output_to_actions(action1).squeeze()
            new_action2 = map_output_to_actions(action2).squeeze()


            # reform action lists, so env can read it properly
            # Lower obs space for step function
            observation, reward, terminated, truncated = env.step(observation, [new_action1, new_action2], t, render_mode)
            reward = reward.clone().detach()
            done = terminated or truncated
            # print(reward, done)
            total_rewardsa.append(reward[0])
            total_rewardsb.append(reward[1])
            if done:
                break
    return  [total_rewardsa, total_rewardsb]

def matchups(agents, n_episodes, env, render_mode = False):
    total_reward = []
    rewards = train([agents[0], agents[1]], env, render_mode, n_episodes)
    total_reward = sum(rewards[0]), sum(rewards[1])
    # if verbose:
    print(total_reward)

def select_action(state, policy_net):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

def load_models(n_agents, save_version):
    Agents = []
    for i in range(n_agents):
        path = f"/Users/christophermao/Desktop/RLModels/{save_version}save_for_agent{i}_policy_net.pt"
        Agents.append(torch.load(path))

# we can test how our model is training by comparing it to past agents, so take WR and rewards
# Test agents by playing it against past agents and take wr and rewards
path1 = f"/Users/christophermao/Desktop/RLModels/2.0.0save_for_agent5_policy_net.pt"
path2 = f"/Users/christophermao/Desktop/RLModels/2.0.0save_for_agent4_policy_net.pt"

modela = torch.load(path1)
modelb = torch.load(path2)
agents = [modela, modelb]

from SimplifiedEnviornment import simple_env
env = simple_env()
# Reset env and get obs length
state, n_obs = env.reset()

matchups(agents, 1, env, render_mode=True)

