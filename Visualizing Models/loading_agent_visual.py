# loading_agent_visual.py
# by christophermao
# 4/16/24

# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm

from envs.Enviornment import GridSoccer
from Graphs import Graph
from Networks import pt_agent

global verbose
verbose = False

global Mtime
Mtime = False

rew_graph = Graph(["Total Reward Graph", "Timesteps", "Reward"])
rew_graph2 = Graph(["Final Reward Graph", "Timesteps", "Reward"])

env = GridSoccer(render_mode='human')
# print(GridSoccer.observation_space, GridSoccer.action_space)

Eps_start = 0.5
Eps_end = 0.05
Eps_decay = 10_000
num_episodes = 10
lr=0.01
model_path = f"/Users/christophermao/Desktop/RLModels/GS-DQN-1.0_policy_net.pt"


dqn = torch.load(model_path)
agent = pt_agent(dqn)

count = 0

for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    total_reward = []
    len_episode = 0

    while True:
        len_episode += 1
        time1 = time.time()
        count += 1
        if verbose:
            print(count)
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

        time2 = time.time()

        time3 = time.time()

        time4 = time.time()

        if Mtime:
            print("Time for select action + env step", time2-time1)
            print("Time for pushing to memory + optimize model", time3-time2)
            print("Time for copying weights to target net", time4-time3)
        total_reward.append(reward)
        if done:
            # node = (count, sum(total_reward))
            # rew_graph.add_node(node)
            # node = (count, reward)
            # rew_graph2.add_node(node)
            print("Episode finished")
            print("Total reward:", sum(total_reward))
            print("Len of episode:", len_episode)
            break


