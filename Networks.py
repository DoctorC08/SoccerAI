# Networks.py
# by christophermao
# 11/13/23

import math
import random
import time

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

global verbose
verbose = False

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

# Allow updates to graph as training progresses
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CircularBuffer(object):
    def __init__(self, buffer_size):
        self.index = -1
        self.buffer_size = buffer_size
        self.data = [0 for _ in range(buffer_size)]
        self.is_full_buffer = False

    def push(self, *args):
        self.index += 1
        if self.index >= self.buffer_size:
            self.index = 0
            self.is_full_buffer = True
        self.data[self.index] = Transition(*args)

    def sample(self, batch_size):
        # print(self.data)
        if self.is_full_buffer:
            return random.sample(self.data, batch_size)
        else:
            return random.sample(self.data[:self.index], batch_size)

    def __len__(self):
        if self.is_full_buffer:
            return self.buffer_size
        else:
            return self.index


class DQN(nn.Module): # Add: Double + Dueling DQN

    def __init__(self, n_observations, n_actions):
        if verbose:
            print("n_obs and n_acts:", n_observations, n_actions)
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 32),
            nn.Mish(),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Linear(32, n_actions)
        )
    def forward(self, x):
        # print(x)
        return self.model(x)

class Agent:
    def __init__(self, n_actions, n_observations, batch_size=25, mem_capacity=1_000, n_agents=10, EPS_START=100, EPS_END=0.20, EPS_DECAY=100_000, LR=1e-2, GAMMA=0.99):
        self.n_actions = n_actions
        self.n_agents = n_agents                      # n_agents is the number of total agents in the enviornment, so updates can roll out every round-robin play through
        self.BATCH_SIZE = batch_size * self.n_agents   # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = GAMMA                       # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = EPS_START                    # EPS_START is the starting value of epsilon
        self.EPS_END = EPS_END                     # EPS_END is the final value of epsilon
        self.EPS_DECAY = EPS_DECAY                 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005                        # TAU is the update rate of the target network
        self.LR = LR                          # LR is the learning rate of the ``AdamW`` optimizer
        # TODO: Try increasing LR to as high as possible
        # Could try cosineannealing for LR

        self.episode_durations = []

        # Declare policy and target nets
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = CircularBuffer(mem_capacity)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("SELECTED ACTION", self.policy_net(state).max(1).indices.view(1, 1))
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # print("RANDOM ACTION")
            return torch.tensor(random.randint(0, self.n_actions - 1), device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # print(transitions)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        if verbose:
            print("batch state len:", len(batch.state))
            print("batch reward len:", len(batch.reward))
            print("batch action len:", len(batch.action))
            # print("batch state", batch.state)
            # print("batch reward", batch.reward)
            # print("batch action", batch.action)

        # make batches a tensor
        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.stack(batch.action, dim=0)
        action_batch = action_batch.view(self.BATCH_SIZE, 1)
        reward_batch = torch.squeeze(torch.stack(batch.reward, dim=0))
        if verbose:
            print("batch state len:", len(state_batch))
            print("batch reward len:", len(reward_batch))
            print("batch state shape", self.policy_net(state_batch).shape)
            print("batch action shape:", action_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        if verbose:
            print("next state values:", next_state_values.shape)
            # print("sample next state values:", next_state_values[:10])
            print("reward batch:", reward_batch.shape)
            # print("sample reward batch:", reward_batch)
            print("state action values:", state_action_values.shape)
            print("expected state action values:", expected_state_action_values.shape)
            # print("sample expected state action values:", expected_state_action_values[:10][:10])
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def convert_list_of_tensors_to_tensor(self, tensor_list):
        list_of_tensors = [torch.tensor(item) for sublist in tensor_list for item in sublist]
        stacked_tensor = torch.stack(list_of_tensors, dim=0)
        return stacked_tensor

    def save_models(self, model_path):
        torch.save(self.policy_net, model_path + "_policy_net.pt")
        torch.save(self.target_net, model_path + "_target_net.pt")


# training and plot duration funcs:
    #
    # def train(self, num_episodes = 600):
    #     for i_episode in range(num_episodes):
    #         print(f"Starting Training for {i_episode} Episode")
    #         # Initialize the environment and get it's state
    #         state = self.env.reset()
    #         state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    #         for t in count():
    #             action = self.select_action(state)
    #             observation, reward, terminated, truncated, _ = self.env.step(action.item())
    #             self.episode_reward = reward
    #             reward = torch.tensor([reward], device=device)
    #             done = terminated or truncated
    #
    #             if terminated:
    #                 next_state = None
    #             else:
    #                 next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    #
    #             # Store the transition in memory
    #             self.memory.push(state, action, next_state, reward)
    #
    #             # Move to the next state
    #             state = next_state
    #
    #             # Perform one step of the optimization (on the policy network)
    #             self.optimize_model()
    #
    #             # Soft update of the target network's weights
    #             # θ′ ← τ θ + (1 −τ )θ′
    #             target_net_state_dict = self.target_net.state_dict()
    #             policy_net_state_dict = self.policy_net.state_dict()
    #             for key in policy_net_state_dict:
    #                 target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
    #             self.target_net.load_state_dict(target_net_state_dict)
    #
    #             if done:
    #                 self.episode_durations.append(t + 1)
    #                 self.plot_durations(reward, show_result=False, )
    #                 break
    #
    #     print('Training complete')
    #     self.plot_durations(100, show_result=True)
    #
    # def plot_durations(self, reward, show_result=False):
    #     plt.figure(1)
    #     durations_t = torch.tensor(reward, dtype=torch.float)
    #
    #     if show_result:
    #         plt.title('Result')
    #     else:
    #         plt.clf()
    #         plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Rewards')
    #     plt.plot(durations_t.numpy())
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())
    #
    #     plt.pause(0.001)
    #     if not show_result:
    #         plt.show()  # Display the plot in the console #


class ptAgent:
    def __init__(self, n_actions, policy_net, target_net, batch_size=25, mem_capacity=1_000, n_agents=10, EPS_START=100, EPS_END=0.05, EPS_DECAY=100_000, LR=1e-2, GAMMA=0.99, steps_done=0):
        self.n_actions = n_actions
        self.n_agents = n_agents                      # n_agents is the number of total agents in the enviornment, so updates can roll out every round-robin play through
        self.BATCH_SIZE = batch_size * self.n_agents   # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = GAMMA                       # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = EPS_START                    # EPS_START is the starting value of epsilon
        self.EPS_END = EPS_END                     # EPS_END is the final value of epsilon
        self.EPS_DECAY = EPS_DECAY                 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005                        # TAU is the update rate of the target network
        self.LR = LR                          # LR is the learning rate of the ``AdamW`` optimizer
        # TODO: Try increasing LR to as high as possible
        # Could try cosineannealing for LR

        self.episode_durations = []

        # Declare policy and target nets
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = CircularBuffer(mem_capacity)

        self.steps_done = steps_done

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("SELECTED ACTION", self.policy_net(state).max(1).indices.view(1, 1))
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # print("RANDOM ACTION")
            return torch.tensor(random.randint(0, self.n_actions - 1), device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # print(transitions)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        if verbose:
            print("batch state len:", len(batch.state))
            print("batch reward len:", len(batch.reward))
            print("batch action len:", len(batch.action))
            # print("batch state", batch.state)
            # print("batch reward", batch.reward)
            # print("batch action", batch.action)

        # make batches a tensor
        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.stack(batch.action, dim=0)
        action_batch = action_batch.view(self.BATCH_SIZE, 1)
        reward_batch = torch.squeeze(torch.stack(batch.reward, dim=0))
        if verbose:
            print("batch state len:", len(state_batch))
            print("batch reward len:", len(reward_batch))
            print("batch state shape", self.policy_net(state_batch).shape)
            print("batch action shape:", action_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        if verbose:
            print("next state values:", next_state_values.shape)
            # print("sample next state values:", next_state_values[:10])
            print("reward batch:", reward_batch.shape)
            # print("sample reward batch:", reward_batch)
            print("state action values:", state_action_values.shape)
            print("expected state action values:", expected_state_action_values.shape)
            # print("sample expected state action values:", expected_state_action_values[:10][:10])
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def convert_list_of_tensors_to_tensor(self, tensor_list):
        list_of_tensors = [torch.tensor(item) for sublist in tensor_list for item in sublist]
        stacked_tensor = torch.stack(list_of_tensors, dim=0)
        return stacked_tensor

    def save_models(self, model_path):
        torch.save(self.policy_net, model_path + "_policy_net.pt")
        torch.save(self.target_net, model_path + "_target_net.pt")
