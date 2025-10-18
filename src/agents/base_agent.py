from abc import ABC, abstractmethod
from gymnasium import spaces
import torch
from dataclasses import dataclass
from typing import Dict, Union

@dataclass
class BaseAgent(ABC):

    state_size: spaces.Space
    action_size: spaces.Space

    device: torch.device
    
    learning_rate: float
    gamma: float
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
    def select_action(self, state: torch.Tensor, is_training: bool = True) -> Union[int, torch.Tensor]:
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        '''
        Must return a dictionary of losses and metrics for the logger
        '''
        pass
    
    # utilities: 

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        pass