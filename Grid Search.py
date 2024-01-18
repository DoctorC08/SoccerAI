# Grid Search.py
# by christophermao
# 12/23/23
import copy
import torch

from main_for_simple_env import matchups, plot_durations, plot_lines, plot_mov_avg_lines, single_player_matchups, single_agent_eval
from SimplifiedEnviornment import simple_env
from Networks import Agent

from tqdm import tqdm as tqdm

global verbose
verbose = False

def create_agents(total_n_actions, n_obs, batch_size=25, mem_capacity=10_000, n_agents=10, EPS_START=100, EPS_END=0.05, EPS_DECAY=100_000, LR=1e-2, GAMMA=0.99):
    # Create an agent
    agent1 = Agent(total_n_actions, n_obs, batch_size, mem_capacity, n_agents, EPS_START, EPS_END, EPS_DECAY, LR, GAMMA)

    Agent1 = [agent1]
    agents = []
    for _ in range(n_agents):
        # Copy agents for num agents needed
        agents.append(copy.deepcopy(Agent1))
    return agents

def round_robin(Agents, extended_file_path_name="", n_try=1):
    n_training_steps = 10_000
    time_to_train = []

    # Create total rewards to track rewards over time
    total_rewards = [[0 for _ in range(n_training_steps)] for _ in range(len(Agents))]
    total_rewards = torch.tensor(total_rewards)

    # Train for 1_000 round-robins
    for i in tqdm(range(n_training_steps), desc=f"Completing {extended_file_path_name}"):
        # save every 1000 round matchups
        if i % n_training_steps-1 == 0 and i != 0:
            rewards = matchups(Agents, 1, env, time_to_train, render_mode=False, performace=True)
            for agent in range(len(Agents)):
                model_path = f"/Users/christophermao/Desktop/RLModels/Grid Search Models/grid_searching:{n_try}.{int(i/500)}_agent_{agent}_{extended_file_path_name}"
                Agents[agent][0].save_models(model_path)
        else:
            rewards = matchups(Agents, 1, env, time_to_train, render_mode=False, performace=False)

        # Add new rewards to total rewards
        for j in range(len(Agents)):
            total_rewards[j][i] = rewards[j]


    # Plot rewards for agents
    sum_agent_scores = torch.tensor([torch.sum(i) for i in total_rewards])
    max_agent_score = total_rewards[torch.argmax(sum_agent_scores)]
    avg_agent_score = torch.tensor([sum(col) / len(total_rewards) for col in zip(*total_rewards)])
    # print(max_agent_score.shape)
    # print(avg_agent_score.shape)
    # print(total_rewards.shape)
    # print(total_rewards)
    plot_lines([max_agent_score, avg_agent_score], [f"{extended_file_path_name} Avg vs Max agent rewards", "Episodes", "Reward"])
    plot_lines(total_rewards, [f"{extended_file_path_name} 10 Agent rewards", "Episodes", "Reward"])

    plot_mov_avg_lines([max_agent_score, avg_agent_score], [f"{extended_file_path_name} Avg vs Max agent rewards", "Episodes", "Reward"], filter=10)
    plot_mov_avg_lines(total_rewards, [f"{extended_file_path_name} 10 Agent rewards", "Episodes", "Reward"], filter=10)
    
def single_agent_training(agent, n_training_steps, extended_file_path_name="", n_try=1, render_mode=False):
    # Create total rewards to track rewards over time
    total_rewards = [0.0 for _ in range(n_training_steps)]
    total_rewards = torch.tensor(total_rewards)
    eval_rewards = [0.0 for _ in range(n_training_steps // 10)]
    eval_rewards = torch.tensor(eval_rewards)
    total_loss = [0.0 for _ in range(n_training_steps)]

    # Train for 1_000 round-robins
    for i in range(n_training_steps):
        # save last training step
        if i % 1_000 == 0 and i != 0:
            if verbose:
                print("saving model at timestep:", i)
            rewards, loss = single_player_matchups(agent, env, render_mode=render_mode)
            model_path = f"/Users/christophermao/Desktop/RLModels/Grid Search Models/single_agent_{n_try}.{i/1000}_{extended_file_path_name}_agent"
            agent.save_models(model_path)
        else:
            rewards, loss = single_player_matchups(agent, env, render_mode=render_mode)
        if i % 10:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (
                        1 - agent.TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            # Eval
            reward = single_agent_eval(agent, env, 8, render_mode=False)
            eval_rewards[i // 10] = sum(reward) / len(reward)

        # Add new rewards to total rewards
        # print(rewards)
        total_rewards[i] = rewards[0]
        total_loss[i] = loss
    if verbose:
        print(total_rewards)
    # Plot training data
    plot_lines(total_rewards, [f"{extended_file_path_name} Agent rewards (raw data)", "Episodes", "Reward"])
    plot_mov_avg_lines(total_rewards, [f"{extended_file_path_name} Agent rewards (avg 50)", "Episodes", "Reward"], filter=50)
    plot_mov_avg_lines(total_rewards, [f"{extended_file_path_name} Agent rewards (avg 100)", "Episodes", "Reward"], filter=100)

    # Plot eval data
    plot_lines(eval_rewards, [f"EVAL: {extended_file_path_name} Agent rewards (raw data)", "Episodes", "Reward"])
    plot_mov_avg_lines(eval_rewards, [f"EVAL: {extended_file_path_name} Agent rewards (avg 10)", "Episodes", "Reward"], filter=10)

    # Plot loss data
    plot_lines(total_loss, [f"{extended_file_path_name} Agent loss (raw data)", "Episodes", "Loss"])
    plot_mov_avg_lines(total_loss, [f"{extended_file_path_name} Agent loss (avg 50)", "Episodes", "Loss"], filter=50)
    plot_mov_avg_lines(total_loss, [f"{extended_file_path_name} Agent loss (avg 100)", "Episodes", "Loss"], filter=100)



# Create an instance of the Agent and enviornment and train the model
env = simple_env()
n_agents = 1
# Reset env and get obs length
state, n_obs = env.reset()

# Get number of actions from gym action space
# n_actions = extract_action_space_numbers(env.action_space)
# print(n_actions)
# n_actions = n_actions[:len(n_actions)//2]
# print(n_actions)
n_actions = 4 # TODO fix this at some point so that it actually calculates it from the env

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
LRs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 5e-3, 5e-4, 5e-5]
gammas = [0.7, 0.85, 0.99]
# LRs = [1e-2, 1e-3, 1e-4, 1e-5]
LRs = [1e-2, 1e-3, 1e-4, 1e-5]
agent = Agent(n_actions, n_obs, EPS_END=1)

# Random agent training for graph comparison
single_agent_training(agent, n_training_steps=10_000, extended_file_path_name="random", render_mode=False)

import threading

threads = []

# Training for different lrs
for lr in LRs:
    for i in tqdm(range(5), desc=f"Training LR:{lr}"):
        agent = Agent(n_actions, n_obs, LR=lr)
        single_agent_training(agent, n_training_steps=10_000, extended_file_path_name=f"LR_{lr}", n_try=i, render_mode=False)

# Graph: Training loss + Eval loss + comparing current vs random or elo scores
# Graph: Avg. and max reward (high level) + 10 lines for each agent (lower level)
# Decrease batch size: 1_000
# re-run same learning rates
# Try lower lrs like 0.01 0.1
# for 1 run for 24 hours


# Graph with moving average
# Matplotlib plotting function: fill between (use transparent and plot from high values to low values

# one agent with stil ball position
# One agent with moving ball
# One agent vs random

# Eval: as often as possible without it being too burdensome
#   Run 8 episodes and take average or plot min avg max

# Plot loss curves, MS of paramaters

# Plot random agent
# Be careful of broadcasting: making sure shapes of tensors are right size

