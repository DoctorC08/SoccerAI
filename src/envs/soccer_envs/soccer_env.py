from src.envs.mod_base_gym_env import modBaseGymEnv
import gymnasium as gym
import soccer_env 
import torch
import numpy as np

class SoccerEnv(modBaseGymEnv):
    metadata = {"render_modes": [None, "human"], "render_fps": 4}
    def __init__(self):
        team_a_size = 2
        team_b_size = 2
        num_players = team_a_size + team_b_size
        width = 100.0
        height = 60.0
        time_step = 0.1
        self.env = soccer_env.SoccerEnv(team_a_size, team_b_size, width, height, time_step)
        self.action_space = gym.spaces.Discrete(6) 

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2 + num_players*4,), dtype=np.float32)

    def change_render_mode(self, new_render_mode):
        self.render_mode = new_render_mode

    def get_size(self):
        return self.size

    def step(self, actions_a, actions_b):
        self.env.step(actions_a, actions_b)
        self.state = self.env.get_state()
        reward = self._calculate_reward()
        terminated = self.env.is_done() # TODO: add terminated vs truncated logic
        return torch.tensor(self.state), reward, False, terminated, {}

    def reset(self):
        self.env.reset()
        return torch.tensor(self.env.get_state()), {}
    
    def render(self):
        #TODO: add rendering 
        pass 

    def close(self):
        pass 