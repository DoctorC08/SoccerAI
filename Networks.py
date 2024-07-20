# Networks.py
# by christophermao
# 11/13/23

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

global verbose
verbose = False

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

# Allow updates to graph as training progresses
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mpu")


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
            if self.index >= 0:
                return self.index
            else:
                return 0

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, fc_layer=16):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(n_observations, affine=False, track_running_stats=True),
            nn.Linear(n_observations, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, fc_layer),
            nn.Mish(),
            nn.Linear(fc_layer, n_actions),
        )

    def exponential_moving_average(self, input, moving_avg, ema_decay=0.99):
        return input * (1 - ema_decay) + ema_decay * moving_avg

    def normalize_inputs(self, input, moving_avg_x, moving_avg_x2, ema_decay=0.99, eps=1e-12):
        # x
        moving_avg_x = self.exponential_moving_average(input, moving_avg_x, ema_decay)
        # x^2
        moving_avg_x2 = self.exponential_moving_average(input**2, moving_avg_x2, ema_decay)
        moving_avg_std = torch.sqrt(moving_avg_x2 - moving_avg_x**2)
        normalized = (input-moving_avg_x) / (moving_avg_std + eps)
        return normalized, moving_avg_x, moving_avg_x2

    def forward(self, x):
        if verbose:
            print("input into nn:", x)
        try:
            x = x.clone().detach()
        except:
            x = torch.tensor(x, dtype=torch.float32)
        match len(x.shape):
            case 1:
                x = x[None]
            case 2:
                pass
            case _:
                raise ValueError(f"expected 1d or 2d input, got {x.shape}")
        return self.model(x)

class Agent:
    def __init__(self, n_actions, n_observations, eps_start=0.9, eps_end=0.05, eps_decay=10_000, lr=1e-6, fc_layer=16, mem_size=10_000):
        # TODO: stack previous obs

        self.loss = 1
        self.n_actions = n_actions
        self.n_agents = 1                      # n_agents is the number of total agents in the enviornment, so updates can roll out every round-robin play through
        self.BATCH_SIZE = 3200 * self.n_agents # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.99                       # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = eps_start                    # EPS_START is the starting value of epsilon
        self.EPS_END = eps_end                     # EPS_END is the final value of epsilon
        self.EPS_DECAY = eps_decay                 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005                        # TAU is the update rate of the target network
        self.LR = lr                          # LR is the learning rate of the ``AdamW`` optimizer

        self.episode_durations = []

        # Declare policy and target nets
        self.policy_net = DQN(n_observations, n_actions, fc_layer=fc_layer).to(device)
        self.target_net = DQN(n_observations, n_actions, fc_layer=fc_layer).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = CircularBuffer(mem_size)

        self.steps_done = 0

    def get_eps(self):
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        return eps

    def select_action(self, state):
        self.policy_net.eval()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                if verbose:
                    print("policy output:", self.policy_net(state))
                    print("argmax func:", torch.argmax(self.policy_net(state)))
                return torch.argmax(self.policy_net(state))
                # return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(random.randint(0, self.n_actions - 1), device=device, dtype=torch.long)

    def select_non_random_action(self, state):
        self.policy_net.eval()
        return torch.argmax(self.policy_net(state))

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        self.policy_net.train()
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                           if s is not None], dim=0)
        if verbose:
            print("batch next state len:", len(batch.next_state))
            print("batch state len:", len(batch.state))
            print("batch reward len:", len(batch.reward))
            print("batch action len:", len(batch.action))

        # make batchese tensor
        # batch.state = torch.tensor(batch.state)
        # print("batch.state:", batch.state)
        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.stack(batch.action, dim=0).unsqueeze(1)
        # reward_batch = torch.squeeze(torch.stack(batch.reward, dim=0))
        reward_batch = torch.tensor(batch.reward)
        if verbose:
            print("batch state len:", len(state_batch))
            print("batch reward len:", len(reward_batch))
            print("batch state shape", self.policy_net(state_batch).shape)
            print("batch action shape:", action_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        if verbose:
            print("state action values shape:", state_action_values.shape)
            print("non final nexxt states len:", len(non_final_next_states))
            # print("non final next states shape:", non_final_next_states.shape)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        if verbose:
            print("next_state_values shape:", next_state_values.shape)
            print("next state values:", next_state_values)
            print("reward batch:", reward_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        if verbose:
            print("next state values:", next_state_values.shape)
            # print("sample next state values:", next_state_values[:10])
            print("reward batch:", reward_batch.shape)
            print("sample reward batch:", reward_batch)
            print("state action values:", state_action_values.shape)
            print("expected state action values:", expected_state_action_values.shape)
            # print("sample expected state action values:", expected_state_action_values[:10][:10])
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.loss = loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, model_path):
        torch.save(self.policy_net, model_path + "_policy_net.pt")


    def convert_list_of_tensors_to_tensor(self, tensor_list):
        list_of_tensors = [torch.tensor(item) for sublist in tensor_list for item in sublist]
        stacked_tensor = torch.stack(list_of_tensors, dim=0)
        return stacked_tensor



class pt_agent:
    def __init__(self, agent_net):
        self.loss = 1
        self.n_agents = 1                       # n_agents is the number of total agents in the enviornment, so updates can roll out every round-robin play through
        self.BATCH_SIZE = 100 * self.n_agents   # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.99                       # GAMMA is the discount factor as mentioned in the previous section

        self.episode_durations = []

        # Declare policy and target nets
        self.policy_net = agent_net
        self.memory = CircularBuffer(10000)

        self.steps_done = 0

    def select_action(self, state):
        return torch.argmax(self.policy_net(state))
