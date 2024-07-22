import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import random
import math

global verbose
verbose = False

# Batch memory
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = [] 
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    # Take a subsection of the batches
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
    
    # store memory 
    # TODO: change datatype for more efficient data storage
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action )
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # clear memory
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
    
# Actor network
class ActorNetwork(nn.Module):
    def __init__(self, input_size, n_actions, alpha, input_dims=1, n_frame_stack=4, fc_config={'conv1': 64, 'conv2': 32, 'linear': 32, 
                                                                          'kernal': 3, 'stride': 3, 'padding': 0}):
        super(ActorNetwork, self).__init__()

        # self.prev_states = [None] * n_frame_stack # keep track of the past n states
        # self.stack_index = n_frame_stack - 1
        # self.n_frame_stack = n_frame_stack

        conv_output_size1 = self.compute_conv_output_size(input_size[0], fc_config)
        conv_output_size2 = self.compute_conv_output_size(input_size[1], fc_config)
        flattened_size = conv_output_size1 * conv_output_size2 * fc_config['conv2']

        self.model = nn.Sequential(
            nn.Conv2d(n_frame_stack, fc_config['conv1'], fc_config['kernal'], fc_config['stride'], fc_config['padding']),
            nn.BatchNorm2d(fc_config['conv1']),
            nn.Mish(),
            nn.Conv2d(fc_config['conv1'], fc_config['conv2'], fc_config['kernal'], fc_config['stride'], fc_config['padding']),
            nn.BatchNorm2d(fc_config['conv2']),
            nn.MaxPool2d(fc_config['kernal'], fc_config['stride']),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(flattened_size, fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # optim
        self.device = torch.device("mps") # declare device
        self.to(self.device) # store class in device

    def compute_conv_output_size(self, size, fc_config):
        size = math.floor((size - fc_config['kernal'] + 2 * fc_config['padding']) / fc_config['stride'] + 1)
        size = math.floor((size - fc_config['kernal'] + 2 * fc_config['padding']) / fc_config['stride'] + 1)
        size = math.floor((size - fc_config['kernal']) / fc_config['stride'] + 1)
        return size

    def forward(self, state, train_mode=True, first_frame=False): # forward propogation
        # if first_frame:
        #     self.prev_states = [state] * self.n_frame_stack
        # else:
        #     self.prev_states[self.stack_index] = state
        #     self.stack_index = (self.stack_index + 1) % self.n_frame_stack

        # # print("prev states:", self.prev_states)
        # stacked_states = torch.stack(self.prev_states, dim=1).to(self.device)

        # if train_mode: # train mode for training
        #     # print("stacked states:", stacked_states.shape)
        #     dist = self.model(stacked_states) # get distributions

        #     dist = Categorical(dist) # sample from distributions
        #     return dist
        # else: # eval mode for evaluation
        #     dist = self.model(stacked_states)
        #     return torch.argmax(dist) # get arg max from dist
        if train_mode: # train mode for training
            self.model.train()
            dist = self.model(state) # get distributions
            dist = Categorical(dist) # sample from distributions
            return dist
        else: # eval mode for evaluation
            self.model.eval()
            dist = self.model(state)
            return torch.argmax(dist)
    
    def load(self, model_path): # load model
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path): # save model
        torch.save(self.model, model_path + "_Actor_PPO.pt")

# Critic Network
# input dims = 4 because [stacks, color, width, height]
class CriticNetwork(nn.Module):
    def __init__(self, input_size, alpha, input_dim=1, n_frame_stack=4, fc_config={'conv1': 64, 'conv2': 32, 'linear': 32, 
                                                                            'kernal': 3, 'stride': 3, 'padding': 0}):
        super(CriticNetwork, self).__init__()

        # self.prev_states = [None] * n_frame_stack # keep track of the past n states
        # self.stack_index = n_frame_stack - 1
        # self.n_frame_stack = n_frame_stack

        conv_output_size1 = self.compute_conv_output_size(input_size[0], fc_config)
        conv_output_size2 = self.compute_conv_output_size(input_size[1], fc_config)
        flattened_size = conv_output_size1 * conv_output_size2 * fc_config['conv2']

        self.model = nn.Sequential(
            nn.Conv2d(n_frame_stack, fc_config['conv1'], fc_config['kernal'], fc_config['stride'], fc_config['padding']),
            nn.BatchNorm2d(fc_config['conv1']),
            nn.Mish(),
            nn.Conv2d(fc_config['conv1'], fc_config['conv2'], fc_config['kernal'], fc_config['stride'], fc_config['padding']),
            nn.BatchNorm2d(fc_config['conv2']),
            nn.MaxPool2d(fc_config['kernal'], fc_config['stride']),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(flattened_size, fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], fc_config['linear']),
            nn.Mish(),
            nn.Linear(fc_config['linear'], 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # optim
        self.device = torch.device("mps") # declare device
        self.to(self.device) # store class in device

    def compute_conv_output_size(self, size, fc_config):
        size = math.floor((size - fc_config['kernal'] + 2 * fc_config['padding']) / fc_config['stride'] + 1)
        size = math.floor((size - fc_config['kernal'] + 2 * fc_config['padding']) / fc_config['stride'] + 1)
        size = math.floor((size - fc_config['kernal']) / fc_config['stride'] + 1)
        return size

    def forward(self, state, first_frame=False): # forward propogation
        # if first_frame:
        #     self.prev_states = [state] * self.n_frame_stack
        # else:
        #     self.prev_states[self.stack_index] = state
        #     self.stack_index = (self.stack_index + 1) % self.n_frame_stack

        # stacked_states = torch.stack(self.prev_states, dim=1).to(self.device)   
           
        # value = self.model(stacked_states)
        # return value
        value = self.model(state)
        return value

    
    def load(self, model_path): # load model
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path): # save model
        torch.save(self.model, model_path + "_Critic_PPO.pt")

class PPOConvAgent:
    def __init__(self, input_size, n_actions, n_frame_stack=4, gamma=0.99, alpha=0.003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, n_epochs=1): # TODO: add paramaters for fc_config
        self.gamma = gamma                      # future discount rewards
        self.policy_clip = policy_clip          # clipping parameter
        self.n_epochs = n_epochs                # n epochs
        self.gae_lambda = gae_lambda            # lambda discount
        self.n_frame_stack = n_frame_stack

        # networks and memory
        self.actor = ActorNetwork(input_size, n_actions, alpha, n_frame_stack=n_frame_stack)
        self.critic = CriticNetwork(input_size, alpha, n_frame_stack=n_frame_stack)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, rewards, done): # store values in memory
        self.memory.store_memory(state, action, probs, vals, rewards, done)

    def save_models(self, name=""): # save models
        self.actor.save_model("/Users/christophermao/Desktop/PPO/" + name)
        self.critic.save_model("/Users/christophermao/Desktop/PPO/" + name)
    
    def load_models(self): # load models
        self.actor.load("/Users/christophermao/Desktop/PPO/")
        self.critic.load("/Users/christophermao/Desktop/PPO/")

    def choose_action(self, observation, train_mode=True, first_frame=False):
        if train_mode: # train mode
            state = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.actor.device)

            with torch.no_grad():
                if first_frame:
                    dist = self.actor(state, first_frame=True)
                    value = self.critic(state, first_frame=True)
                else:
                    dist = self.actor(state)
                    value = self.critic(state)
                action = dist.sample()
                

            probs = dist.log_prob(action).item()
            action = action.item()
            value = value.item()

            return action, probs, value
        else: # eval mdoe
            state = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.actor.device)

            with torch.no_grad():
                action = self.actor(state)
            action = action.item()

            return action
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, dones_arr, batches = self.memory.generate_batches() # get batches

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1): # loop through "memories"
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k]) # evaluate advantage at timestep t
                    discount *= self.gamma*self.gae_lambda # evaluate discount
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device) 

            values = torch.tensor(values, dtype=torch.float32).to(self.actor.device)
            # print("batches:", batches)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)

                # print("states in update shape:", states.shape)
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions) 
                prob_ratio = new_probs.exp() / old_probs.exp() # ratio to clamp 
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
        return total_loss



