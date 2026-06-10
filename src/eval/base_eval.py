from abc import ABC, abstractmethod
import torch 
from typing import List, Dict
from src.envs.mod_base_gym_env import modBaseGymEnv

class BaseEval(ABC):
    def __init__(self, render_evals: bool, env: modBaseGymEnv, get_action, get_metrics) -> None:
        self.render_evals = render_evals
        self.env = env
        self.get_action = get_action
        self.get_metrics = get_metrics
        
    @abstractmethod
    def eval(self, log=True):
        pass 

