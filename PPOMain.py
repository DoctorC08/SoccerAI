import gymnasium as gym
import numpy as np
from PPO import PPOAgent
from tqdm import tqdm

env = gym.make('CartPole-v1')
N = 20
batch_size = 5
n_epochs = 10
alpha = 0.0003
agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
              n_epochs=n_epochs, input_dims=env.observation_space.shape)

n_games = 300
best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in tqdm(range(n_games)):
    observation, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        obs, reward, terminated, truncated, _ = env.step(action)
        n_steps += 1
        score += reward
        done = terminated or truncated
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = obs
    score_history.append(score)
    if (len(score_history) > 100):
        avg_score = np.mean(score_history[-100])
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 
          'time_steps', n_steps, 'learning_steps', learn_iters)

# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history, figure_file)