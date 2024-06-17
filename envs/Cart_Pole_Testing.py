# Cart_Pole_Testing.py
# by christophermao
# 4/23/24

import torch
import time
from tqdm import tqdm
import wandb
import numpy as np
import gymnasium as gym

from Networks import Agent

global verbose
verbose = False

global Mtime
Mtime = False




def train(config):
    env = gym.make('CartPole-v1')

    Eps_start = config.Eps_start
    Eps_end = config.Eps_end
    Eps_decay = config.Eps_decay
    lr = config.Lr
    fc_layer = config.fc_layer
    mem_size = config.mem_size
    name = f"GS-DQN-1.0"
    model_path = f"/Users/christophermao/Desktop/RLModels/{name}"

    agent = Agent(2, 4, eps_start=Eps_start, eps_end=Eps_end, eps_decay=Eps_decay,
                  lr=lr, fc_layer=fc_layer, mem_size=mem_size)

    count = 0

    while True:
        state, _ = env.reset()
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
            obs, reward, terminated, truncated, _ = env.step(np.array(action))
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

            if count % 128:
                agent.optimize_model()
                loss.append(agent.loss)

            time3 = time.time()

            if count % config.save_interval == 0:
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

        sum_total_rew = sum(total_reward)
        wandb.log({
            "len_episodes:": len_episode,
            "total_reward": sum_total_rew,
            "average_reward": sum_total_rew / len_episode,
            "sum_loss": sum(loss),
            "average_loss": sum(loss) / len(loss),
            "eps": agent.get_eps(),
        })

        if count > config.timesteps:
            break

    agent.save_model(model_path)

    # get average reward for 100 episodes
    final_reward = []
    for i in range(100):
        total_reward = []
        obs, _ = env.reset()
        while True:
            action = agent.select_non_random_action(obs)
            obs, reward, terminated, truncated, _ = env.step(np.array(action))
            done = terminated or truncated
            total_reward.append(reward)
            if done:
                break
        final_reward.append(sum(total_reward))

    return sum(final_reward) / 100




# init wandb
wandb.login()

# Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "score"
    },
    "parameters": {
        "Eps_decay": {
            "distribution": "int_uniform",
            "max": 20000,
            "min": 1000
        },
        "Eps_start": {
            "distribution": "uniform",
            "max": 0.99,
            "min": 0.25,
        },
        "Eps_end": {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.025,
        },
        "Lr": {
            "distribution": "uniform",
            "max": 0.1,
            "min": 1e-07,
        },
        "timesteps": {
            "distribution": "int_uniform",
            "max": 100_000,
            "min": 10_00,
        },
        "fc_layer": {
            "distribution": "int_uniform",
            "max": 128,
            "min": 8,
        },
        "mem_size": {
            "distribution": "int_uniform",
            "max": 50_000,
            "min": 500,
        },
        "save_interval": {
            "distribution": "int_uniform",
            "max": 2_000,
            "min": 500,
        }
    },
}

def main():
    wandb.init(project="CartPole")
    score = train(wandb.config)
    print(score)
    wandb.log({"final": score})

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CartPole")

wandb.agent(sweep_id, function=main)

