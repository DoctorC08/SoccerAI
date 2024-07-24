import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo


import numpy as np
from PPOConvNet import PPOConvAgent
from tqdm import tqdm
import wandb
import time

wandb.login()


def train(config):
    env = gym.make("ALE/Tetris-v5", obs_type="grayscale", frameskip=1, render_mode='rgb_array')
    env = AtariPreprocessing(env, screen_size=84, scale_obs=True)
    env = RecordVideo(env, 
                      video_folder="/Users/christophermao/Documents/GitHub/SoccerAI/Videos", 
                      episode_trigger=lambda x: x % 50 == 0, 
                      video_length=500)
    # env.metadata['render_fps'] = 60
    
    N = config.N                            # update agent every N steps: 20
    batch_size = config.batch_size          # Batch size: 5
    n_epochs = config.n_epochs              # n epochs: 10
    alpha = config.lr                       # lr: 0.0003
    n_games = config.n_games                # n games: 300
    # n_frame_stack = config.n_frame_stack    # stacked frames 
    n_frame_stack = 1

    fc_config={
        'conv1': config.conv1, 
        'conv2': config.conv1, 
        'linear': config.linear, 
        'kernal': config.kernal, 
        'stride': config.stride, 
        'padding': config.padding
    }
    # fc_config={'conv1': 64, 'conv2': 32, 'linear': 32, 'kernal': 3, 'stride': 3, 'padding': 0}


    stack_index = 1


    agent = PPOConvAgent(input_size=env.observation_space.shape, n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
              n_epochs=n_epochs, n_frame_stack=n_frame_stack, fc_config=fc_config)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    agent.actor.train()
    agent.critic.train()
    for i in tqdm(range(n_games)):
        observation, _ = env.reset()
        done = False
        score = 0
        loss = []

        observation = [observation] * n_frame_stack

        while not done:
            action, prob, val = agent.choose_action(observation)

            obs, reward, terminated, truncated, _ = env.step(action)
            n_steps += 1
            score += reward
            done = terminated or truncated
            agent.remember(observation, action, prob, val, reward, done)

            observation[stack_index] = obs
            stack_index = (stack_index + 1) % n_frame_stack

            if n_steps % N == 0: # update
                loss.append(agent.learn())
                learn_iters += 1
            # observation = obs
            # first_frame = False
        

        score_history.append(score)

        if (len(score_history) > 100):
            avg_score = np.mean(score_history[-100])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(name="Tetris")

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 
            'time_steps', n_steps, 'learning_steps', learn_iters)
        
        wandb.log({
            'Score': score, 
            'Average Score': avg_score,
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
            "min": 20
        },
        "batch_size": {
            "distribution": "int_uniform",
            "max": 100,
            "min": 20,
        },
        "n_epochs": {
            "distribution": "int_uniform",
            "max": 20,
            "min": 5,
        },
        "lr": {
            "distribution": "uniform",
            "max": 0.001,
            "min": 1e-07,
        },
        "n_games": {
            "distribution": "int_uniform",
            "max": 10000,
            "min": 1000,
        },
        # "n_frame_stack": {
        #     "distribution": "int_uniform",
        #     "max": 8,
        #     "min": 2,
        # }, 
        "conv1": {
            "distribution": "int_uniform",
            "max": 64,
            "min": 16,
        },
        "conv2": {
            "distribution": "int_uniform",
            "max": 32,
            "min": 8,
        },
        "linear": {
            "distribution": "int_uniform",
            "max": 64,
            "min": 16,
        },
        "kernal": {
            "distribution": "int_uniform",
            "max": 4,
            "min": 3,
        },
        "stride": {
            "distribution": "int_uniform",
            "max": 4,
            "min": 3,
        },
        "padding": {
            "distribution": "int_uniform",
            "max": 1,
            "min": 0,
        },
    },
}

def main():
    wandb.init()
    score = train(wandb.config)
    wandb.log({"score": score})

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Tetris")

wandb.agent(sweep_id, function=main, count=20)

# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history, figure_file)