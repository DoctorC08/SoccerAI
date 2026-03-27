from src.envs.mod_base_gym_env import modBaseGymEnv
from src.envs.soccer_envs.soccer_renderer import SoccerRenderer
import gymnasium as gym
from src.envs.soccer_envs import soccer_sim
import torch
import numpy as np

class SoccerEnv(modBaseGymEnv):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 4}
    def __init__(self, team_a_size: int = 2, team_b_size: int = 2, width: float = 100.0, height: float = 60.0, time_step: float = 0.1, goal_size: float = 20, render_mode: str = None) -> None:
        self.team_a_size = team_a_size
        self.team_b_size = team_b_size

        # if human render mode force team size to be 1v0
        if render_mode == "human":
            self.team_a_size = 1
            self.team_b_size = 0

        self.num_players = self.team_a_size + self.team_b_size
        self.width = width
        self.height = height
        self.time_step = time_step
        self.render_mode = render_mode
        self.goal_size = goal_size
        self.env = soccer_sim.SoccerEnv(self.team_a_size, self.team_b_size, width, height, time_step, goal_size, kf=20.0, fric=0.85, bmw = 0.5, pmw = 0.2)
        self.action_space = gym.spaces.Discrete(6) 

        self.next_action = None 

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2 + self.num_players*4,), dtype=np.float32)
        if render_mode in ["human", "rgb_array"]:
            self.renderer = SoccerRenderer(h=600, w=1000, ui_width=400, display_mode=render_mode)
        
    def change_render_mode(self, new_render_mode):
        self.render_mode = new_render_mode


    def step(self, actions):
        if self.next_action is not None:
            actions = self.next_action
        
        # print("actions:", actions)
        self.env.step(actions)
        state = self.env.get_state()
        reward = self.env.get_rewards()
        terminated = self.env.is_done() 
        
        if self.render_mode == "human":
            self.next_action = self.render()
        if self.render_mode == "rgb_array":
            return torch.tensor(state), reward, terminated, False, {}, self.render()
        return torch.tensor(state), reward, terminated, False, {}

    def reset(self):
        self.env.reset()
        return torch.tensor(self.env.get_state()), {}
    
    def render(self):
        if self.render_mode == "human":
            action = self.renderer.draw(self.env.get_state(), self.env.get_rewards())
            return action
        return self.renderer.draw(self.env.get_state(), self.env.get_rewards())

    def close(self):
        pass 

if __name__ == '__main__':
    env = SoccerEnv(team_a_size=1, team_b_size=0, render_mode="human")
    obs, info = env.reset()
    action = [0]
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs, reward, terminated)
        if terminated:
            break