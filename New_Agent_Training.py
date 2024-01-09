# New_Agent_Training.py
# by christophermao
# 12/30/23
from SimplifiedEnviornment import simple_env
from main_for_simple_env import create_agents, matchups, plot_durations


global verbose
verbose = False



# Create an instance of the Agent and enviornment and train the model
env = simple_env()
n_agents = 10
# Reset env and get obs length
state, n_obs = env.reset()

if verbose:
    print("state:", state, " n_obs:", n_obs)

# Get number of actions from gym action space
# n_actions = extract_action_space_numbers(env.action_space)
# print(n_actions)
# n_actions = n_actions[:len(n_actions)//2]
# print(n_actions)
n_actions = 8 # TODO fix this at some point so that it actually calculates it from the env

# Convert list to num of possible outputs
if verbose:
    print("total_n_actions: ", n_actions)
    print("total action + obs", n_actions, n_obs)

# Create the agents
Agents = create_agents(n_agents, n_actions, n_obs)
time_to_train = []

# Train for n round-robins
for i in range(100):
    matchups(Agents, 1, env, time_to_train, render_mode=False)
    print(f"Finished {i + 1} matchups")
    if i % 10 == 0:
        plot_durations(time_to_train, ["Time to train", "Time (sec)", "Episodes"], show_result=True)
        for agent in range(len(Agents)):
            model_path = f"/Users/christophermao/Desktop/RLModels/2.2.{int(i/10)}save_for_agent{agent}"
            Agents[agent][0].save_models(model_path)

# TODO: we can test how our model is training by comparing it to past agents, so take WR and rewards
# 40 is intrestingr5
