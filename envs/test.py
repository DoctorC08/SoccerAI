# test.py
# by christophermao
# 4/4/24
import torch
import time
from tqdm import tqdm
import wandb

from Enviornment import GridSoccer
from Networks import Agent
from Graphs import Graph

global verbose
verbose = False

global Mtime
Mtime = False



# init wandb
wandb.login()


# run = wandb.init(
#     # Set the project where this run will be logged
#     project="SoccerAI",
#     # Track hyperparameters and run metadata
#     config={
#         "Eps_start": Eps_start,
#         "Eps_end": Eps_end,
#         "Eps_decay": Eps_decay,
#         "episodes": num_episodes,
#         "lr": lr,
#         "name": name,
#     },
#     # mode="disabled",
# )

def train(config):
    env = GridSoccer()

    Eps_start = config.Eps_start
    Eps_end = config.Eps_end
    Eps_decay = config.Eps_decay
    num_episodes = config.episodes
    lr = config.Lr
    name = f"GS-DQN-1.0"
    model_path = f"/Users/christophermao/Desktop/RLModels/{name}"

    agent = Agent(4, 4, eps_start=Eps_start, eps_end=Eps_end, eps_decay=Eps_decay, lr=lr)

    count = 0
    final_rews = []

    for episode in tqdm(range(num_episodes)):
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

        sum_total_rew = sum(total_reward)
        wandb.log({
            "len_episodes:": len_episode,
            "total_reward": sum_total_rew,
            "average_reward": sum_total_rew / len_episode,
            "sum_loss": sum(loss),
            "average_loss": sum(loss) / len(loss),
            "eps": agent.get_eps(),
        })
        final_rews.append(sum(total_reward))

    agent.save_model(model_path)

    # average total rew of past 100 eps
    return sum(final_rews[-101:-1]) / 100




# 2: Define the search space
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
            "min": 1e-05,
        },
        "episodes": {
            "distribution": "int_uniform",
            "max": 4000,
            "min": 1000,
        },
    },
}

def main():
    wandb.init(project="SoccerAI")
    score = train(wandb.config)
    wandb.log({"final": score})


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="SoccerAI")

wandb.agent(sweep_id, function=main, count=10)
