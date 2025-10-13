import gymnasium as gym
import numpy as np
from PPO import PPOAgent
from tqdm import tqdm
import wandb
import torch
import time

wandb.login()

def train(config):

    env = gym.make('CartPole-v1')
    N = config.N                            # update agent every N steps: 20
    batch_size = config.batch_size          # Batch size: 5
    n_epochs = config.n_epochs              # n epochs: 10
    alpha = config.lr                       # lr: 0.0003
    n_games = config.n_games                # n games: 300
    fc_layers = config.fc_layers            # neurons in layer

    time1 = time.time()
    agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
              n_epochs=n_epochs, input_dims=env.observation_space.shape, fc_layer=fc_layers)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    time2 = time.time()
    for i in tqdm(range(n_games)):
        observation, _ = env.reset()
        done = False
        score = 0
        loss = []

        time3 = 0
        time4 = 0
        time5 = 0
        time6 = 0
        time7 = 0

        while not done:
            time3 += time.time()
            action, prob, val = agent.choose_action(observation)

            time4 += time.time()

            obs, reward, terminated, truncated, _ = env.step(action)
            
            time5 += time.time()

            n_steps += 1
            score += reward
            done = terminated or truncated
            agent.remember(observation, action, prob, val, reward, done)
            
            time6 += time.time()
            if n_steps % N == 0:
                loss.append(agent.learn())
                learn_iters += 1
            
            time7 += time.time()
            observation = obs
        
        # print("Timing:")
        # print(time2-time1)
        # print(time4-time3)
        # print(time5-time4)
        # print(time6-time5)
        # print(time7-time6)

        score_history.append(score)
        if (len(score_history) > 100):
            avg_score = np.mean(score_history[-100])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

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
    for i in range(20):
        total_reward = []
        obs, _ = env.reset()
        while True:
            # obs = torch.tensor(np.array(obs), dtype=torch.float).to(agent.actor.device)
            action = agent.actor(torch.from_numpy(obs), train_mode=False)
            print("Action: " + action)
            # action = torch.squeeze(action).item() 
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward.append(reward)
            if done:
                break
        final_reward.append(sum(total_reward))

    return sum(final_reward) / 20

    

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
    # wandb.init(project="CartPole", mode="disabled")
    score = train(wandb.config)
    wandb.log({"score": score})


# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CartPole")

wandb.agent(sweep_id, function=main, count=1)

# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history, figure_file)