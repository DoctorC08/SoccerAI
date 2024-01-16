# main_for_simple_env.py
# by christophermao
# 12/12/23

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

from Networks import Agent, ptAgent
from SimplifiedEnviornment import simple_env

import copy
import pandas as pd
import time

import numpy as np

from tqdm import tqdm

from torch.optim import AdamW

# Define Verbose
global verbose
verbose = False
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

# Allow updates to graph as training progresses
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Def train
def train(Agents, env, render_mode, num_episodes=600): # Agents is a list of Agents
    # take training time data
    rewards0 = []
    rewards1 = []
    time0 = []
    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []
    time6 = []

    for i_episode in range(num_episodes):
        # print(f"Starting Training for {i_episode} Episode")
        # Initialize the environment and get it's obs
        observation, _ = env.reset()
        # observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        observation = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)

        for t in count():
            # print every 1,000th timestep
            # if t % 1_000 == 0:
            #     print("timestep:", t)
            if verbose:
                print("observation:", observation, "len obs:", len(observation[0]))
                print("Agents:", Agents)
            # Get actions from agents
            start_time = time.time()
            action1 = Agents[0][0].select_action(observation).squeeze()
            action2 = Agents[1][0].select_action(observation).squeeze()

            observation = observation.squeeze(0)

            if verbose:
                print("action1:", action1)
                print("action2:", action2)

            # Convert actions from numbers into lists
            new_action1 = map_output_to_actions(action1).squeeze() # TODO change this final list if I ever change action shape
            new_action2 = map_output_to_actions(action2).squeeze()


            # reform action lists, so env can read it properly
            if verbose:
                print("OHE actions:", action1)
                print("new action1:", new_action1)
                print("new action2:", new_action2)
            end_time = time.time()
            time0.append(end_time-start_time)

            start_time = time.time()
            # Lower obs space for step function
            observation, reward, terminated, truncated = env.step(observation, [new_action1, new_action2], t, render_mode)
            reward = reward.clone().detach()
            reward = add_outer_dimension(reward)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                next_state = observation.clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0)

            end_time = time.time()
            time1.append(end_time - start_time)

            start_time = time.time()

            # Store the transition in memory
            Agents[0][0].memory.push(observation, action1.clone().detach(), next_state, reward[0].clone().detach())
            Agents[1][0].memory.push(observation, action2.clone().detach(), next_state, reward[1].clone().detach())

            end_time = time.time()
            time2.append(end_time - start_time)

            start_time = time.time()
            # Move to the next state
            observation = next_state
            if t % 40:
                end_timea = time.time()
                # print("agent:", Agents)
                for i in range(len(Agents)):
                    # print(Agents[i][0])
                    if verbose:
                        print("Soft update for:", Agents[i][0])
                        # print("Agent num:", Agents.index(agent))
                        # Perform one step of the optimization (on the policy network)
                        print("timestep:", t)


                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = Agents[i][0].target_net.state_dict()
                    policy_net_state_dict = Agents[i][0].policy_net.state_dict()

                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * Agents[i][0].TAU + target_net_state_dict[key] * (
                                    1 - Agents[i][0].TAU)
                    Agents[i][0].target_net.load_state_dict(target_net_state_dict)

                time6.append(time.time() - end_timea)
                rewards0.append(reward[0])
                rewards1.append(reward[1])

                end_time = time.time()
                time3.append(end_time - start_time)

            if done:

                # plot_durations(rewards0, show_result=False)
                # print("Finished an episode")
                break
    # print("\n\nTime for selecting actions:")
    # print("Max time:", max(time0))
    # print("total time:", sum(time0))
    # print("avg time:", sum(time0) / len(time0))
    #
    # print("\n\nTime for env.step:")
    # print("Max time:", max(time1))
    # print("total time:", sum(time1))
    # print("avg time:", sum(time1) / len(time1))
    #
    # print("\n\nTime for storing obs:")
    # print("Max time:", max(time2))
    # print("total time:", sum(time2))
    # print("avg time:", sum(time2) / len(time2))
    #
    # print("\n\nTime for optimization step:")
    # print("Max time:", max(time3))
    # print("total time:", sum(time3))
    # print("avg time:", sum(time3) / len(time3))
    #
    # # print("\nTime for optimizing model:")
    # # print("Max time:", max(time4))
    # # print("total time:", sum(time4))
    # # print("avg time:", sum(time4) / len(time4))
    #
    # # print("\nTime for soft update a:")
    # # print("Max time:", max(time5))
    # # print("total time:", sum(time5))
    # # print("avg time:", sum(time5) / len(time5))
    #
    # print("\nTime for soft update:")
    # print("Max time:", max(time6))
    # print("total time:", sum(time6))
    # print("avg time:", sum(time6) / len(time6))


    # print('\n\nTraining complete')
    # plot_durations(rewards0, "rewards 0", show_result=False)
    # plot_durations(rewards1, "rewards 1", show_result=False)
    return [sum(rewards0), sum(rewards1)], time6

def single_agent_train(agent, env, render_mode): # Agents is a list of Agents
    # take training time data
    rewards0 = []
    rewards1 = []

    # print(f"Starting Training for {i_episode} Episode")
    # Initialize the environment and get it's obs
    observation, _ = env.reset()
    # observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    observation = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)

    for t in count():
        if verbose:
            print("observation:", observation, "len obs:", len(observation[0]))
            print("Agents:", agent)
        # Get actions from agents
        action = agent.select_action(observation).squeeze()

        observation = observation.squeeze(0)

        if verbose:
            print("action1:", action)

        # Convert actions from numbers into lists
        new_action = action # TODO change this final list if I ever change action shape

        # reform action lists, so env can read it properly
        if verbose:
            print("OHE actions:", action)
            print("new action1:", new_action)

        # Lower obs space for step function
        observation, reward, terminated, truncated = env.step(observation, [new_action], t, render_mode)
        reward = reward.clone().detach()
        reward = add_outer_dimension(reward)
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = observation.clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0)


        # Store the transition in memory
        agent.memory.push(observation, action.clone().detach(), next_state, reward[0].clone().detach())

        # Move to the next state
        observation = next_state

        rewards0.append(reward[0])
        rewards1.append(reward[1])

        if done:

            # plot_durations(rewards0, show_result=False)
            # print("Finished an episode")
            break
    # print("\n\nTime for selecting actions:")
    # print("Max time:", max(time0))
    # print("total time:", sum(time0))
    # print("avg time:", sum(time0) / len(time0))
    #
    # print("\n\nTime for env.step:")
    # print("Max time:", max(time1))
    # print("total time:", sum(time1))
    # print("avg time:", sum(time1) / len(time1))
    #
    # print("\n\nTime for storing obs:")
    # print("Max time:", max(time2))
    # print("total time:", sum(time2))
    # print("avg time:", sum(time2) / len(time2))
    #
    # print("\n\nTime for optimization step:")
    # print("Max time:", max(time3))
    # print("total time:", sum(time3))
    # print("avg time:", sum(time3) / len(time3))
    #
    # # print("\nTime for optimizing model:")
    # # print("Max time:", max(time4))
    # # print("total time:", sum(time4))
    # # print("avg time:", sum(time4) / len(time4))
    #
    # # print("\nTime for soft update a:")
    # # print("Max time:", max(time5))
    # # print("total time:", sum(time5))
    # # print("avg time:", sum(time5) / len(time5))
    #
    # print("\nTime for soft update:")
    # print("Max time:", max(time6))
    # print("total time:", sum(time6))
    # print("avg time:", sum(time6) / len(time6))


    # print('\n\nTraining complete')
    # plot_durations(rewards0, "rewards 0", show_result=False)
    # plot_durations(rewards1, "rewards 1", show_result=False)
    return [sum(rewards0), sum(rewards1)]

def plot_durations(reward, names, show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(reward, dtype=torch.float)

    if show_result:
        title = f"{names[0]}"
        plt.title(title)
    plt.xlabel(names[2])
    plt.ylabel(names[1])
    plt.plot(reward_t.numpy())
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if not show_result:
        plt.show()  # Display the plot in the console

def plot_lines(data, labels):
    # for i in range(len(data)):
    #     plt.plot(data[i])
    plt.plot(data)
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.show()

def plot_mov_avg_lines(data, labels, filter=100):
    new_data = moving_average(data, n=filter)
    plt.plot(new_data)

    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.show()

def moving_average(a, n=3): # Credit: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def performance_matrix(data, title=None):
    # Creates a performance matrix for a set of agents with names.
    matrix = np.array([[0 for _ in range(10)] for _ in range(10)], dtype=float) # TODO if i change n agents

    # Populate the matrix with values from the dictionary
    for row, col in data:
        matrix[row][col] = data[(row, col)]

    if title:
        plt.title(title)
    # print(matrix)
    plt.matshow(matrix)
    plt.colorbar()
    plt.show()

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

def extract_action_space_numbers(action_space):
    action_space_numbers = []
    for agent_action_space in action_space:
        for action_dim_space in agent_action_space:
            action_space_numbers.append(action_dim_space.n)
    return action_space_numbers

def reshape_tensors_to_scalar(input_tensors):
    reshaped_tensors = []
    for t in input_tensors:
        # Reshape the tensor to zero dimensions
        reshaped_tensor = t.view(-1)
        reshaped_tensors.append(reshaped_tensor)
    return tuple(reshaped_tensors)

def add_outer_dimension(tensor_tuple):
    return tuple(t.unsqueeze(0) for t in tensor_tuple)

def discrete_to_one_hot_vector(one_hot_vector, list_length):
    if verbose:
        print("ohe value", one_hot_vector)
    # Convert the index to a binary list
    binary_list = [0] * list_length
    binary_list[one_hot_vector] = 1

    return binary_list

def create_agents(n_agents, total_n_actions, n_obs):
    # Create an agent
    agent1 = [Agent(total_n_actions, n_obs)]

    agents = []
    for _ in range(n_agents):
        # Copy agents for num agents needed
        agents.append(copy.deepcopy(agent1))
    return agents

def get_agents(n_agents, n_actions, start_path, end_path1_policy_net, end_path2_target_net, steps_done=0):
    Agents = []
    for i in range(n_agents):
        new_agent = [ptAgent(n_actions, torch.load(f"{start_path}{i}{end_path1_policy_net}"),
                            torch.load(f"{start_path}{i}{end_path2_target_net}"), EPS_END=0.2, steps_done=steps_done)]
        Agents.append(new_agent)
    return Agents



# Define matchups and create performance matrix
def matchups(agents, n_episodes, env, time_to_train, render_mode = False, performace = True):
    n_agents = len(agents)
    data = {}
    total_rewards = [0 for _ in range(len(agents))]

    # print(render_mode)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            start_time = time.time()
            # print("Playing agent", i, "and agent", j)
            rewards, time6 = train([agents[i], agents[j]], env, render_mode, n_episodes)
            end_time = time.time()
            data[(i, j)] = rewards[0].reshape(1, 1)
            data[(j, i)] = rewards[1].reshape(1, 1)
            time_to_train.append(end_time-start_time)
            total_rewards[i] += rewards[0]
            total_rewards[j] += rewards[1]

            # sum_rewards0.append(rewards[0])
            # sum_rewards1.append(rewards[1])
            # print("Sum rewards:", rewards)
    start_time = time.time()
    for i in range(n_agents):
        agents[i][0].optimize_model()
    # print("time for optimization steps:", time.time() - start_time)
    # plot_durations(time6, ["Time for soft udpate", "Time (sec)", "Episodes"], show_result=True)
    # plot_durations(sum_rewards0, ["Sum Rewards0", "Episodes", "Rewards"], show_result=True)
    # plot_durations(sum_rewards1, ["Sum Rewards0", "Episodes", "Rewards"], show_result=True)

    # print("data", data)
    if performace:
        performance_matrix(data)

    return torch.tensor(total_rewards)

def single_player_matchups(agent, env, render_mode=False):
    # Loop for n_episodes:
    rewards = single_agent_train(agent, env, render_mode)
    agent.optimize_model()

    return rewards[0].clone().detach()
