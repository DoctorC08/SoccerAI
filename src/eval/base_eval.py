from abc import ABC, abstractmethod
import torch 
from typing import List, Dict, Tuple
from src.envs.mod_base_gym_env import modBaseGymEnv

class BaseEval(ABC):
    def __init__(self, render_evals: bool, env: modBaseGymEnv, get_action, get_metrics, fps=5, eval_freq=100) -> None:
        self.render_evals = render_evals
        self.env = env
        self.get_action = get_action
        self.get_metrics = get_metrics
        self.fps = fps
        self.eval_freq = eval_freq
        self.n_eps = 0
        
    @abstractmethod
    def eval(self, log=True) -> Tuple[Dict[str, int], float]:
        pass 

