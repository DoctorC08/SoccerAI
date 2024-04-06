import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch

from Networks import Agent
from Enviornment import env

import copy
import pandas as pd

# Define Verbose
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

# Def Test:
def test(Agents, env, render_mode, num_episodes=600): # Agents is a list of Agents
    rewards0 = []
    rewards1 = []

    for i_episode in range(num_episodes):
        print(f"Starting Training for {i_episode} Episode")
        # Initialize the environment and get it's obs
        observation, _ = env.reset()
        # observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        observation = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)

        for t in count():
            # print every 1,000th timestep
            if t % 1_000 == 0:
                print("timestep:", t)
            # Get actions from agents
            action1 = []
            action2 = []
            for j in range(len(Agents)):
                if j < len(Agents) - 1:
                    for i in range(len(Agents[j])):
                        action1.append(Agents[0][i].select_action(observation))
                else:
                    for i in range(len(Agents[j])):
                        action2.append(Agents[1][i].select_action(observation))


            observation = observation.squeeze(0)
            # for t in action1:
            action1 = reshape_tensors_to_scalar(action1)
            action2 = reshape_tensors_to_scalar(action2)

            # Convert actions from numbers into lists
            new_action1 = map_output_to_actions(action1, [4, 3, 2, 3]) # TODO change this final list if I ever change action shape
            new_action2 = map_output_to_actions(action2, [4, 3, 2, 3])

            # reform action lists, so env can read it properly
            new_action1 = torch.tensor([[num[0] for num in new_action1], [num[1] for num in new_action1]])
            new_action2 = torch.tensor([[num[0] for num in new_action2], [num[1] for num in new_action2]])

            # Lower obs space for step function
            observation, reward, terminated, truncated = env.step(observation, [new_action1, new_action2], t, render_mode)
            reward = reward.clone().detach()
            reward = add_outer_dimension(reward)
            done = terminated or truncated

            if done:

                # plot_durations(rewards0, show_result=False)
                print("Finished an episode")
                break
    print('Training complete')
    plot_durations(rewards0, "rewards 0", show_result=True)
    plot_durations(rewards1, "rewards 1", show_result=True)
    return reward

# Def train
def train(Agents, env, render_mode, num_episodes=600): # Agents is a list of Agents
    rewards0 = []
    rewards1 = []

    for i_episode in range(num_episodes):
        print(f"Starting Training for {i_episode} Episode")
        # Initialize the environment and get it's obs
        observation, _ = env.reset()
        # observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        observation = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)

        for t in count():
            # print every 1,000th timestep
            if t % 1_000 == 0:
                print("timestep:", t)
            if verbose:
                print("observation:", observation, "len obs:", len(observation[0]))
            # Get actions from agents
            action1 = []
            action2 = []
            for j in range(len(Agents)):
                if j < len(Agents) - 1:
                    for i in range(len(Agents[j])):
                        action1.append(Agents[0][i].select_action(observation))
                else:
                    for i in range(len(Agents[j])):
                        action2.append(Agents[1][i].select_action(observation))


            observation = observation.squeeze(0)
            # for t in action1:
            if verbose:
                print("pre-reshape action2:", action1)
                print("pre-reshape action2:", action1)
            action1 = reshape_tensors_to_scalar(action1)
            action2 = reshape_tensors_to_scalar(action2)
            if verbose:
                print("action1:", action1)
                print("action2:", action2)

            # Convert actions from numbers into lists
            new_action1 = map_output_to_actions(action1, [4, 3, 2, 3]) # TODO change this final list if I ever change action shape
            new_action2 = map_output_to_actions(action2, [4, 3, 2, 3])

            # reform action lists, so env can read it properly
            if verbose:
                print("new action1:", new_action1)
                print("new action2:", new_action2)
            new_action1 = torch.tensor([[num[0] for num in new_action1], [num[1] for num in new_action1]])
            new_action2 = torch.tensor([[num[0] for num in new_action2], [num[1] for num in new_action2]])
            if verbose:
                print("action1:", new_action1)
                print("action2:", new_action2)

            # Lower obs space for step function
            observation, reward, terminated, truncated = env.step(observation, [new_action1, new_action2], t, render_mode)
            reward = reward.clone().detach()
            reward = add_outer_dimension(reward)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = observation.clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0)

            total_actions = [16, 9, 4, 9]

            # Check if it's agent type 1 or two
            # print("len agents:", len(Agents))
            # print("agent:", agent)
            for agent in range(len(Agents[0])):
                # Store the transition in memory
                if verbose:
                    print(f"storing in {agent}th agent", torch.tensor(action1[agent]))
                    print(f"storing in {agent}th agent", torch.tensor(next_state))
                    print(f"storing in {agent}th agent", torch.tensor(next_state).shape)
                Agents[0][agent].memory.push(observation, torch.tensor(one_hot_to_binary(action1[agent], total_actions[agent])), next_state, torch.tensor(reward[0]))

            for agent in range(len(Agents[1])):
                # Store the transition in memory
                if verbose:
                    print(f"storing in {agent}th agent", torch.tensor(action1[agent - (len(Agents) // 2)]))
                    print(f"storing in {agent}th agent", torch.tensor(next_state))
                    print(f"storing in {agent}th agent", torch.tensor(next_state).shape)
                Agents[1][agent].memory.push(observation, torch.tensor(one_hot_to_binary(action2[agent], total_actions[agent])), next_state, torch.tensor(reward[1]))

            # Move to the next state
            observation = next_state
            for i in range(len(Agents)):
                for agent in Agents[i]:
                    if verbose:
                        print("Optimization for:", agent)
                        print("Agent num:", Agents.index(agent))
                        # Perform one step of the optimization (on the policy network)
                        print("timestep:", t)
                    agent.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = agent.target_net.state_dict()
                    policy_net_state_dict = agent.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (
                                    1 - agent.TAU)
                    agent.target_net.load_state_dict(target_net_state_dict)
            rewards0.append(reward[0])
            rewards1.append(reward[1])

            if done:

                # plot_durations(rewards0, show_result=False)
                print("Finished an episode")
                break
    print('Training complete')
    plot_durations(rewards0, "rewards 0", show_result=True)
    plot_durations(rewards1, "rewards 1", show_result=True)
    return reward

def plot_durations(reward, name, show_result=False): #TODO plot multiple graphs for differenet ganets, or change it to a matrix
    plt.figure(1)
    reward_t = torch.tensor(reward, dtype=torch.float)

    if show_result:
        title = f"Result: {name}"
        plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.plot(reward_t.numpy())
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if not show_result:
        plt.show()  # Display the plot in the console

# Creates a performance matrix for a set of agents with names.
def performance_matrix(data, agent_names):

  # Create an empty DataFrame
  df = pd.DataFrame(index=agent_names, columns=agent_names)

  # Fill the DataFrame with performance values
  for agent_pair, metrics in data.items():
    agent1, agent2 = agent_pair
    df.loc[agent_names[agent1], agent_names[agent2]] = metrics

  # Return the DataFrame
  return df

def map_output_to_actions(output, constraint):
    output = list(output)
    if verbose:
        print("pre action values:", output)
    # print("output: ", output, " constraint", constraint)
    return_values = []
    for i in range(len(output)):
        if output[i] % constraint[i] == 0:
            return_values.append([output[i] // constraint[i] - 1, constraint[i] - 1])
        else:
            return_values.append([output[i] // constraint[i], output[i] % constraint[i]])
    if verbose:
        print("return action values:", torch.tensor(return_values))
    return torch.tensor(return_values)


def extract_action_space_numbers(action_space):
    action_space_numbers = []
    for agent_action_space in action_space:
        for action_dim_space in agent_action_space:
            action_space_numbers.append(action_dim_space.n)
    return action_space_numbers

def reshape_tensors_to_scalar(input_tensors):
    reshaped_tensors = []
    for t in input_tensors:
        # Reshape the tensor to zero dimensions
        reshaped_tensor = t.view(-1)
        reshaped_tensors.append(reshaped_tensor)
    return tuple(reshaped_tensors)

def add_outer_dimension(tensor_tuple):
    return tuple(t.unsqueeze(0) for t in tensor_tuple)

def one_hot_to_binary(one_hot_vector, list_length):
    # Extract the index of the non-zero element
    action = torch.argmax(one_hot_vector).item()

    # Convert the index to a binary list
    binary_list = [0] * list_length
    binary_list[action] = 1

    return binary_list

def create_agents(n_agents, total_n_actions, n_obs):

    agent1a = Agent(total_n_actions[0], n_obs)
    agent1b = Agent(total_n_actions[1], n_obs)
    agent1c = Agent(total_n_actions[2], n_obs)
    agent1d = Agent(total_n_actions[3], n_obs)

    Agent1 = [agent1a, agent1b, agent1c, agent1d]
    agents = []
    for _ in range(n_agents):
        agents.append(copy.deepcopy(Agent1))
    return agents

# Define matchups and create performance matrix
def matchups(agents, n_episodes, env, render_mode = False):
    n_agents = len(agents)
    data = {}
    print(render_mode)
    for i in range(n_agents):
        for j in range(n_agents - 1):
            if j < i:
                print("Playing agent", i, "and agent", j, f"for {n_episodes} episodes")
                rewards = train([agents[i], agents[j]], env, render_mode, n_episodes)
                data[(i, j)] = rewards
            else:
                print("Playing agent", i, "and agent", j + 1, f"for {n_episodes} episodes")
                rewards = train([agents[i], agents[j + 1]], env, render_mode, n_episodes)
                data[(i, j + 1)] = rewards
    agent_names = [f"Agent {i}" for i in range(len(agents))]
    matrix = performance_matrix(data, agent_names)
    print(matrix)
    print(data)


# Create an instance of the Agent and enviornment and train the model
env = env()
n_agents = 20
# Reset env and get obs length
state, n_obs = env.reset()

if verbose:
    print("state:", state, " n_obs:", n_obs)

# Get number of actions from gym action space
n_actions = extract_action_space_numbers(env.action_space)
n_actions = n_actions[:len(n_actions)//2]

# Convert list to num of possible outputs
total_n_actions = []
for i in range(len(n_actions) // 2):
    total_n_actions.append(n_actions[i] ** 2)
if verbose:
    print("total_n_actions: ", total_n_actions)
model_path = "/Users/christophermao/Desktop/RLModels"

# Train for 100,000 round-robins
for i in range(100_000):
    # Initialize agents for different action sets
    Agents = create_agents(n_agents, total_n_actions, n_obs)
    matchups(Agents, 1, env, render_mode=False)

    plt.ioff()
    plt.show()
    if i % 1000:
        for i in range(len(Agents)):
            for agent in Agents[i]:
                torch.save(agent.state_dict(), model_path)
