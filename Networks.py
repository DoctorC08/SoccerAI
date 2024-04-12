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


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 16),
            nn.Mish(),
            nn.Linear(16, 32),
            nn.Mish(),
            nn.Linear(32, 16),
            nn.Mish(),
            nn.Linear(16, n_actions),
        )
    def forward(self, x):
        if verbose:
            print("input into nn:", x)
        try:
            return self.model(x)
        except:
            return self.model(torch.tensor(x, dtype=torch.float32))
        # except TypeError:
        #     input = []
        #     for value in x.values:
        #         print(value)
        #         for item in value:
        #             input.append(item)
        #     print("forward input:", input)
        #     return self.model(input)

class Agent:
    def __init__(self, n_actions, n_observations, eps_start=0.9, eps_end=0.05, eps_decay=10_000, lr=1e-6):
        self.loss = 1
        self.n_actions = n_actions
        self.n_agents = 1                      # n_agents is the number of total agents in the enviornment, so updates can roll out every round-robin play through
        self.BATCH_SIZE = 100 * self.n_agents # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.99                       # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = eps_start                    # EPS_START is the starting value of epsilon
        self.EPS_END = eps_end                     # EPS_END is the final value of epsilon
        self.EPS_DECAY = eps_decay                 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005                        # TAU is the update rate of the target network
        self.LR = lr                          # LR is the learning rate of the ``AdamW`` optimizer

        self.episode_durations = []

        # Declare policy and target nets
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net2 = DQN(n_observations, n_actions).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def get_eps(self):
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        return eps

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
                if verbose:
                    print("policy output:", self.policy_net(state))
                    print("argmax func:", torch.argmax(self.policy_net(state)))
                return torch.argmax(self.policy_net(state))
                # return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(random.randint(0, self.n_actions - 1), device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
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


    def convert_list_of_tensors_to_tensor(self, tensor_list):
        list_of_tensors = [torch.tensor(item) for sublist in tensor_list for item in sublist]
        stacked_tensor = torch.stack(list_of_tensors, dim=0)
        return stacked_tensor



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
