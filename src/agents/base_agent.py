from abc import ABC, abstractmethod
from gymnasium import spaces
import torch
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class BaseAgent(ABC):

    state_size: spaces.Space
    action_size: spaces.Space

    device: torch.device
    
    learning_rate: float
    grad_clip: float
    
    train_start: int
    train_step: int = 0
    
    training: bool = True

    def __post_init__(self):
        self.policy_network = None
        self.optimizer = None
        
        self._setup_model()

    @abstractmethod
    def _setup_model(self) -> None:
        pass

    @abstractmethod
    def select_action(self, state: torch.Tensor, is_training: bool = True) -> Tuple[int, torch.Tensor]: 
        pass

    @abstractmethod
    def update(self, state, action, reward, done, log_prob=None, identifier=None) -> Dict[str, float]:
        '''
        Method that will update the models inside of the agent. 
        Must return a dictionary of losses and metrics for the logger

        Args: 
            state: list of sampled states
            action: ^^
            reward: ^^
            done: ^^
            log_prob: ^^
            identifier: specific identification key if multiple updates need to happen on the same step. 
                Will return metrics with identifier to distinguish multiple updates during same timestep
        '''
        pass
    
    # utilities: 

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        pass

    @abstractmethod
    def get_models(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_loss_fns(self) -> torch.nn.Module: 
        pass