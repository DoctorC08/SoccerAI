import gymnasium as gym
from abc import ABC, abstractmethod

class modBaseGymEnv(ABC, gym.Env):
    def change_render_mode(self, new_render_mode):
        self.render_mode = new_render_mode