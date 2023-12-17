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

from Networks import Agent
from SimplifiedEnviornment import simple_env

import copy
import pandas as pd
import time

import numpy as np

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
    # TODO: take training time data
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

            # Convert action1/2 to binary
            # action1 = discrete_to_one_hot_vector(action1, 8) # TODO Change this if i change num actions
            # action2 = discrete_to_one_hot_vector(action2, 8)


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

            if terminated:
                next_state = None
            else:
                next_state = observation.clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0)

            end_time = time.time()
            time1.append(end_time - start_time)

            start_time = time.time()

            # Store the transition in memory
            if verbose:
                pass
                # print(f"storing in {agent}th agent", torch.tensor(action1[agent]))
                # print(f"storing in {agent}th agent", torch.tensor(next_state))
                # print(f"storing in {agent}th agent", torch.tensor(next_state).shape)
            Agents[0][0].memory.push(observation, action1.clone().detach(), next_state, reward[0].clone().detach())

            # Store the transition in memory
            if verbose:
                pass
                # print(f"storing in {agent}th agent", torch.tensor(action1[agent - (len(Agents) // 2)]))
                # print(f"storing in {agent}th agent", torch.tensor(next_state))
                # print(f"storing in {agent}th agent", torch.tensor(next_state).shape)
            Agents[1][0].memory.push(observation, action2.clone().detach(), next_state, reward[1].clone().detach())

            end_time = time.time()
            time2.append(end_time - start_time)

            start_time = time.time()
            # Move to the next state
            observation = next_state
            if t % 100:
                # print("agent:", Agents)
                for i in range(len(Agents)):
                    # print(Agents[i][0])
                    if verbose:
                        print("Optimization for:", Agents[i][0])
                        # print("Agent num:", Agents.index(agent))
                        # Perform one step of the optimization (on the policy network)
                        print("timestep:", t)
                    start_timea = time.time()
                    Agents[i][0].optimize_model()
                    end_timea = time.time()
                    time4.append(end_timea-start_timea)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = Agents[i][0].target_net.state_dict()
                    policy_net_state_dict = Agents[i][0].policy_net.state_dict()

                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * Agents[i][0].TAU + target_net_state_dict[key] * (
                                    1 - Agents[i][0].TAU)
                    Agents[i][0].target_net.load_state_dict(target_net_state_dict)

                    time6.append(time.time()-end_timea)
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

    # print("\n\nTime for env.step:")
    # print("Max time:", max(time1))
    # print("total time:", sum(time1))
    # print("avg time:", sum(time1) / len(time1))

    # print("\n\nTime for storing obs:")
    # print("Max time:", max(time2))
    # print("total time:", sum(time2))
    # print("avg time:", sum(time2) / len(time2))

    # print("\n\nTime for optimization step:")
    # print("Max time:", max(time3))
    # print("total time:", sum(time3))
    # print("avg time:", sum(time3) / len(time3))

    # print("\nTime for optimizing model:")
    # print("Max time:", max(time4))
    # print("total time:", sum(time4))
    # print("avg time:", sum(time4) / len(time4))

    # print("\nTime for soft update a:")
    # print("Max time:", max(time5))
    # print("total time:", sum(time5))
    # print("avg time:", sum(time5) / len(time5))

    # print("\nTime for soft update:")
    # print("Max time:", max(time6))
    # print("total time:", sum(time6))
    # print("avg time:", sum(time6) / len(time6))

    # print('\n\nTraining complete')
    # plot_durations(rewards0, "rewards 0", show_result=False)
    # plot_durations(rewards1, "rewards 1", show_result=False)
    return [sum(rewards0), sum(rewards1)]

def plot_durations(reward, names, show_result=False): #TODO plot multiple graphs for differenet ganets, or change it to a matrix
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

def performance_matrix(data):
    # Creates a performance matrix for a set of agents with names.
    # # Create an empty DataFrame
    # df = pd.DataFrame(index=agent_names, columns=agent_names)
    #
    # # Fill the DataFrame with performance values
    # for agent_pair, metrics in data.items():
    #     agent1, agent2 = agent_pair
    #     df.loc[agent_names[agent1], agent_names[agent2]] = metrics
    #
    # # Return the DataFrame
    # return df
    matrix = np.array([[0 for _ in range(10)] for _ in range(10)], dtype=float) #TODO if i change n agents

    # Populate the matrix with values from the dictionary
    for row, col in data:
        matrix[row][col] = data[(row, col)]

    print(matrix)
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
    agent1 = Agent(total_n_actions, n_obs)

    Agent1 = [agent1]
    agents = []
    for _ in range(n_agents):
        # Copy agents for num agents needed
        agents.append(copy.deepcopy(Agent1))
    return agents

# Define matchups and create performance matrix
def matchups(agents, n_episodes, env, sum_rewards0, sum_rewards1, time_to_train, render_mode = False):
    n_agents = len(agents)
    data = {}

    # print(render_mode)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            start_time = time.time()
            print("Playing agent", i, "and agent", j)
            rewards = train([agents[i], agents[j]], env, render_mode, n_episodes)
            end_time = time.time()
            data[(i, j)] = rewards[0].reshape(1, 1)
            data[(j, i)] = rewards[1].reshape(1, 1)
            time_to_train.append(end_time-start_time)
            # sum_rewards0.append(rewards[0])
            # sum_rewards1.append(rewards[1])
            # print("Sum rewards:", rewards)
    plot_durations(time_to_train, ["Time to train", "Time (sec)", "Episodes"], show_result=True)
    # plot_durations(sum_rewards0, ["Sum Rewards0", "Episodes", "Rewards"], show_result=True)
    # plot_durations(sum_rewards1, ["Sum Rewards0", "Episodes", "Rewards"], show_result=True)

    agent_names = [f"Agent {i}" for i in range(len(agents))]
    # print("data", data)
    performance_matrix(data)


# Create an instance of the Agent and enviornment and train the model
env = simple_env()
n_agents = 10
# Reset env and get obs length
state, n_obs = env.reset()

if verbose:
    print("state:", state, " n_obs:", n_obs)

# Get number of actions from gym action space
# n_actions = extract_action_space_numbers(env.action_space)
# print(n_actions)
# n_actions = n_actions[:len(n_actions)//2]
# print(n_actions)
n_actions = 8 # TODO fix this at some point so that it actually calculates it from the env

# Convert list to num of possible outputs
if verbose:
    print("total_n_actions: ", n_actions)
    print("total action + obs", n_actions, n_obs)


# Create the agents
Agents = create_agents(n_agents, n_actions, n_obs)
sum_rewards0 = []
sum_rewards1 = []
time_to_train = []

# Train for 1_000 round-robins
for i in range(100):
    matchups(Agents, 1, env, sum_rewards0, sum_rewards1, time_to_train, render_mode=False)
    print(f"Finished {i} matchups")
    if i % 10:
        matchups(Agents, 1, env, sum_rewards0, sum_rewards1, time_to_train, render_mode=True)
        for agent in range(len(Agents)):
            model_path = f"/Users/christophermao/Desktop/RLModels/1.{i/10}save_for_agent{agent}"
            Agents[agent][0].save_models(model_path)

