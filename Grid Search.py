# Grid Search.py
# by christophermao
# 12/23/23
import copy

from main_for_simple_env import matchups
from SimplifiedEnviornment import simple_env
from Networks import Agent

import tqdm as tqdm

global verbose
verbose = False

def create_agents(total_n_actions, n_obs, batch_size, mem_capacity, n_agents, EPS_START, EPS_END, EPS_DECAY, LR, GAMMA):
    # Create an agent
    agent1 = Agent(total_n_actions, n_obs, batch_size, mem_capacity, n_agents, EPS_START, EPS_END, EPS_DECAY, LR, GAMMA)

    Agent1 = [agent1]
    agents = []
    for _ in range(n_agents):
        # Copy agents for num agents needed
        agents.append(copy.deepcopy(Agent1))
    return agents

def round_robin(Agents, extended_file_path_name=""):
    time_to_train = []
    # Train for 1_000 round-robins
    for i in range(50):
        matchups(Agents, 1, env, time_to_train, render_mode=False)
        print(f"Finished {i + 1} matchups")
        if i % 20 == 0:
            for agent in range(len(Agents)):
                model_path = f"/Users/christophermao/Desktop/RLModels/grid_searching:1.{int(i/10)}_agent_{agent}_{extended_file_path_name}"
                Agents[agent][0].save_models(model_path)

# Create an instance of the Agent and enviornment and train the model
env = simple_env()
n_agents = 10
# Reset env and get obs length
state, n_obs = env.reset()

# Get number of actions from gym action space
# n_actions = extract_action_space_numbers(env.action_space)
# print(n_actions)
# n_actions = n_actions[:len(n_actions)//2]
# print(n_actions)
n_actions = 8 # TODO fix this at some point so that it actually calculates it from the env

# Convert list to num of possible outputs
if verbose:
    print("state:", state, " n_obs:", n_obs)
    print("total_n_actions: ", n_actions)
    print("total action + obs", n_actions, n_obs)


# Create the agents
# base agent: n_actions, n_observations, batch_size=200, mem_capacity=1_000, n_agents=10, EPS_START=100, EPS_END=0.05, EPS_DECAY=10_000, LR=1e-6, GAMMA=0.99
# Test baseline:
# Agents = create_agents(n_agents, n_actions, n_obs)
# round_robin(Agents)

batch_sizes = [100, 200, 300]
mem_capacities = [500, 1_000, 5_000]
n_agents = [2, 6, 10, 16]
eps_starts = [80, 90, 100]
eps_ends = [0.0, 0.05, 0.01]
eps_decays = [1_000, 5_000, 10_000]
LRs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]
gammas = [0.7, 0.85, 0.99]

# # Figure out how to do this better:
# for batch_size in range(len(batch_sizes)):
#     for mem_capacity in range(len(mem_capacities)):
#         for eps_start in range(len(eps_starts)):
#             for eps_end in range(len(eps_ends)):
#                 for eps_decay in range(len(eps_decays)):
#                     for LR in range(len(LRs)):
#                         for gamma in range(len(gammas)):
#                             create_agents(n_actions, n_obs, batch_size, mem_capacity, eps_start, eps_end, eps_decay, LR, gamma)


for lr in tqdm(range(len(LRs))):
    Agents = create_agents(n_actions, n_obs, LR=LRs)
    round_robin(Agents, extended_file_path_name=f"LR: {lr}")
