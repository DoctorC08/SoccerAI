from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

from src.agents.base_agent import BaseAgent

@dataclass
class PolicyAgent(BaseAgent, ABC):
    entropy_coef: float = 0.01
    n_epochs: int = 1

    @abstractmethod
    def find_value(self, state: torch.Tensor) -> torch.Tensor:
        pass

    # Override parent select_action method to change return type
    @abstractmethod
    def select_action(self, state: torch.Tensor, is_training: bool = True) -> Tuple[int, torch.Tensor]:
        pass

    # Override parent update method to change parameters 
    @abstractmethod
    def update(self, states, returns, advantages, log_prob=None, identifier=None) -> Dict[str, float]:
        pass
