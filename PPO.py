# import math
# import random
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

global verbose
verbose = False

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = [] 
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indicies)
        batches = [indicies[i : i + self.batch_size] for i in batch_start]
        
        return np.array(self.states), \
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action )
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
    
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc_dims=64):
        super(ActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("mps") 
        self.to(self.device)

    def forward(self, state):
        dist = self.model(state)
        dist = Categorical(dist)

        return dist
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.model, model_path + "_Actor_PPO.pt")

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc_dims=64):
        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("mps") 
        self.to(self.device)

    def forward(self, state):
        value = self.model(state)
        return value
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.model, model_path + "_Critic_PPO.pt")

class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, horizon=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, rewards, done):
        self.memory.store_memory(state, action, probs, vals, rewards, done)

    def save_models(self):
        self.actor.save_model("/Users/christophermao/Desktop/PPO/")
        self.critic.save_model("/Users/christophermao/Desktop/PPO/")
    
    def load_models(self):
        self.actor.load("/Users/christophermao/Desktop/PPO/")
        self.critic.load("/Users/christophermao/Desktop/PPO/")

    def choose_action(self, observation):
        state = torch.tensor(np.array(observation), dtype=torch.float).to(self.actor.device)
        
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item() 
        action = torch.squeeze(action).item() 
        value = torch.squeeze(value).item() 

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values, dtype=torch.float32).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()



