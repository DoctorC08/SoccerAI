import gymnasium as gym
import numpy as np
from PPO import PPOAgent
from tqdm import tqdm
import wandb
import torch
import multiprocessing as mp

wandb.login()

def train(config, device, run_id):
    with wandb.init(project="CartPole", config=config, id=run_id, resume="allow"):
        env = gym.make('CartPole-v1')
        N = config.N
        batch_size = config.batch_size
        n_epochs = config.n_epochs
        alpha = config.lr
        n_games = config.n_games
        fc_layers = config.fc_layers

        agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
                  n_epochs=n_epochs, input_dims=env.observation_space.shape, fc_layer=fc_layers)

        best_score = env.reward_range[0]
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in tqdm(range(n_games)):
            observation, _ = env.reset()
            done = False
            score = 0
            loss = []
            while not done:
                action, prob, val = agent.choose_action(observation)
                obs, reward, terminated, truncated, _ = env.step(action)
                n_steps += 1
                score += reward
                done = terminated or truncated
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0:
                    loss.append(agent.learn())
                    learn_iters += 1
                observation = obs
            score_history.append(score)
            if (len(score_history) > 100):
                avg_score = np.mean(score_history[-100])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models(name=str(i))

            wandb.log({
                'Score': score, 
                'Average Score': avg_score,
                'Time steps': n_steps, 
                'Learning steps': learn_iters, 
                'Total Loss': sum(loss),
                'Average Loss': (sum(loss) / n_steps),
            })
        
        final_reward = []
        for i in range(30):
            total_reward = []
            obs, _ = env.reset()
            while True:
                action = agent.choose_action(obs, train_mode=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward.append(reward)
                if done:
                    break
            final_reward.append(sum(total_reward))

        final_score = sum(final_reward) / 30
        wandb.log({"Final Score": final_score})
        return final_score


def worker(sweep_id, device):
    def train_with_wandb():
        with wandb.init() as run:
            return train(run.config, device, run.id)
    
    wandb.agent(sweep_id, function=train_with_wandb, count=1)


def run_sweep(sweep_id, num_runs):
    device = torch.device("mps")
    
    processes = []
    for _ in range(num_runs):
        p = mp.Process(target=worker, args=(sweep_id, device))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": 'Final Score'
        },
        "parameters": {
            "N": {
                "distribution": "int_uniform",
                "max": 100,
                "min": 5
            },
            "batch_size": {
                "distribution": "int_uniform",
                "max": 25,
                "min": 5,
            },
            "n_epochs": {
                "distribution": "int_uniform",
                "max": 20,
                "min": 5,
            },
            "lr": {
                "distribution": "uniform",
                "max": 0.1,
                "min": 1e-07,
            },
            "n_games": {
                "distribution": "int_uniform",
                "max": 100,
                "min": 20,
            },
            "fc_layers": {
                "distribution": "int_uniform",
                "max": 128,
                "min": 8,
            },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="CartPole")
    
    num_processes = 4  
    run_sweep(sweep_id, num_processes)