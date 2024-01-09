# ptAgent_training.py
# by christophermao
# 12/30/23

from Networks import ptAgent
from SimplifiedEnviornment import simple_env
from main_for_simple_env import get_agents, matchups, plot_durations

n_actions = 8 #TODO: take from env or change if i change n actions
n_agents = 10

start_path = "/Users/christophermao/Desktop/RLModels/2.2.29save_for_agent"
end_path1 = "_policy_net.pt"
end_path2 = "_target_net.pt"
# Create the agents
Agents = get_agents(n_agents, n_actions, start_path, end_path1, end_path2, steps_done=40*9*500)

# create the env
env = simple_env()

time_to_train = []
# Train for n round-robins
for i in range(500, 10_000):
    if i % 100 == 0:
        matchups(Agents, 1, env, time_to_train, render_mode=False, performace=True)
        for agent in range(len(Agents)):
            model_path = f"/Users/christophermao/Desktop/RLModels/2.2.{int(i/10)}save_for_agent{agent}"
            Agents[agent][0].save_models(model_path)
        if i % 500 == 0:
            plot_durations(time_to_train, ["Time to train", "Time (sec)", "Episodes"], show_result=True)
    else:
        matchups(Agents, 1, env, time_to_train, render_mode=False, performace=False)
    print(f"Finished {i + 1} matchups")

