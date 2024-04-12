# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm
import wandb

from envs.Enviornment import GridSoccer
from Networks import Agent
from Graphs import Graph

global verbose
verbose = False

global Mtime
Mtime = False

rew_graph = Graph(["Total Reward Graph", "Timesteps", "Reward"])
rew_graph2 = Graph(["Final Reward Graph", "Timesteps", "Reward"])

env = GridSoccer()
# print(GridSoccer.observation_space, GridSoccer.action_space)

# init wandb
wandb.login()
Eps_start = 0.5
Eps_end = 0.05
Eps_decay = 10_000
num_episodes = 10000
lr=0.001
name = f"GS-DQN-1.0"
model_path = f"/Users/christophermao/Desktop/RLModels/{name}"


agent = Agent(4, 4, eps_start=Eps_start, eps_end=Eps_end, eps_decay=Eps_decay, lr=lr)


run = wandb.init(
    # Set the project where this run will be logged
    project="SoccerAI",
    # Track hyperparameters and run metadata
    config={
        "Eps_start": Eps_start,
        "Eps_end": Eps_end,
        "Eps_decay": Eps_decay,
        "episodes": num_episodes,
        "lr": lr,
        "name": name,
    },
    # mode="disabled",
)

count = 0

for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    # print(episode)
    total_reward = []
    loss = []
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
        agent.memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
        state = next_state

        time2 = time.time()

        agent.optimize_model()
        loss.append(agent.loss)

        time3 = time.time()

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (1 - agent.TAU)
        agent.target_net.load_state_dict(target_net_state_dict)

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
            break

    wandb.log({
        "len_episodes:": len_episode,
        "total_reward": sum(total_reward),
        "average_reward": sum(total_reward) / len_episode,
        "sum_loss": sum(loss),
        "average_loss": sum(loss) / len(loss),
        "eps": agent.get_eps(),
    })

agent.save_model(model_path)
