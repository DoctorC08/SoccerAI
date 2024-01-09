import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, combinations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import Agent, ptAgent
from SimplifiedEnviornment import simple_env

import copy
import pandas as pd
import time

import numpy as np

from tqdm import tqdm

from torch.optim import AdamW

from main_for_simple_env import performance_matrix

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
    print("total_reward:", total_reward)

def matchups2(agents, env, render_mode = False):
    rewards = train([agents[0], agents[1]], env, render_mode, 1)
    return rewards


def select_action(state, policy_net):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

def load_models(n_agents, save_version):
    Agents = []
    for i in range(n_agents):
        path = f"/Users/christophermao/Desktop/RLModels/{save_version}save_for_agent{i}_policy_net.pt"
        Agents.append(torch.load(path))

def test_models(file_path_start, file_path_mid, file_path_end, num_players, versions, times_tested_per_matchup, testing_subject=""):
    # prep data
    data = {}


    for j in versions:
        for i in range(num_players):
            for k in range(i + 1, num_players):
                data[(i, k)] = 0
                data[(k, i)] = 0
        for t in range(times_tested_per_matchup):
            for n in range(num_players):
                for i in range(n + 1, num_players):
                    path1 = f"{file_path_start}{j}{file_path_mid}{n}{file_path_end}"
                    path2 = f"{file_path_start}{j}{file_path_mid}{i}{file_path_end}"
                    agents = [torch.load(path1), torch.load(path2)]
                    agents2 = [torch.load(path2), torch.load(path1)]

                    rewards = matchups2(agents, env, render_mode=True)
                    data[(i, n)] += int(sum(rewards[0]))
                    data[(n, i)] += int(sum(rewards[1]))
                    # print(sum(rewards[0]), sum(rewards[1]))


                    rewards = matchups2(agents2, env)
                    data[(i, n)] += int(sum(rewards[0]))
                    data[(n, i)] += int(sum(rewards[1]))

        performance_matrix(data, title=f"{testing_subject} for Version {j} total rewards from games")
        print(f"{testing_subject} for Version {j} total rewards from games")



# we can test how our model is training by comparing it to past agents, so take WR and rewards
# Test agents by playing it against past agents and take wr and rewards
# 5 LR:0.001
# 3 vs 2 LR:0.0001
# 9 vs 8 LR: 1e-05 (looks uncertain)
# 1e-06  first-last save all agents just go straight left
# 1e-07 all saves agents just go to right and down
LRs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 5e-3, 5e-4, 5e-5]
path1 = f"/Users/christophermao/Desktop/RLModels/Grid Search Models/grid_searching:1.10_agent_7_LR:0.0001_policy_net.pt"
path2 = f"/Users/christophermao/Desktop/RLModels/Grid Search Models/grid_searching:1.10_agent_6_LR:0.0001_policy_net.pt"

modela = torch.load(path1)
modelb = torch.load(path2)
agents = [modela, modelb]

from SimplifiedEnviornment import simple_env

env = simple_env()
# Reset env and get obs length
state, n_obs = env.reset()

# matchups(agents, 1, env, render_mode=True)
# LRs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 5e-3, 5e-4, 5e-5]
LRs = [1e-4]
for i in LRs:
    test_models("/Users/christophermao/Desktop/RLModels/Grid Search Models/grid_searching:1.", "_agent_", f"_LR:{i}_policy_net.pt", 10, [19], 5, f"{i}")


# Graph with moving average
# Matplotlib plotting function: fill between (use transparent and plot from high values to low values

# one agent with stil ball position
# One agent with moving ball
# One agent vs random
