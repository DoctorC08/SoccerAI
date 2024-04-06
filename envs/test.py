# test.py
# by christophermao
# 4/4/24
import gym
import torch
from tqdm import tqdm

from Enviornment import GridSoccer
from Networks import Agent
from Graphs import Graph

global verbose
verbose = False

rew_graph = Graph
env = GridSoccer()
# print(GridSoccer.observation_space, GridSoccer.action_space)
agent = Agent(4, 4)


num_episodes = 100
for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()

    count = 0
    while True:
        count += 1
        if verbose:
            print("state:", state)
        action = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = obs

        if verbose:
            print("memory push values")
            print(state)
            print(action)
            print(reward)
            print(next_state)
            print(done)
        agent.memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
        state = next_state

        agent.optimize_model()

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (1 - agent.TAU)
        agent.target_net.load_state_dict(target_net_state_dict)

        node = (count, reward)
        rew_graph.add_node(node)

rew_graph.display()
