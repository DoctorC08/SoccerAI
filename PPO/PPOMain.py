import gymnasium as gym
import numpy as np
from PPO import PPOAgent
from tqdm import tqdm
import wandb
import torch

wandb.login()



def train(config, device):

    env = gym.make('CartPole-v1')
    N = config.N                            # update agent every N steps: 20
    batch_size = config.batch_size          # Batch size: 5
    n_epochs = config.n_epochs              # n epochs: 10
    alpha = config.lr                       # lr: 0.0003
    n_games = config.n_games                # n games: 300
    fc_layers = config.fc_layers            # neurons in layer


    agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
              n_epochs=n_epochs, input_dims=env.observation_space.shape, fc_layer=fc_layers).to(device)

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

        # print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 
        #     'time_steps', n_steps, 'learning_steps', learn_iters)
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
            # action = torch.squeeze(action).item() 
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward.append(reward)
            if done:
                break
        final_reward.append(sum(total_reward))

    return sum(final_reward) / 30

    

# Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": 'Score'
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

def main():
    wandb.init()
    device = torch.device("mps")
    score = train(wandb.config, device)
    wandb.log({"score": score})

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CartPole")

wandb.agent(sweep_id, function=main, count=20)

# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history, figure_file)