from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from ..base_agent import BaseAgent

@dataclass
class ValueAgent(BaseAgent, ABC):

    epsilon: float
    epsilon_decay: float
    epsilon_min: float

    batch_size: int
    memory_size: int
    update_target_freq: int

    target_network: torch.nn.Module = None


    @abstractmethod
    def calculate_value(self, state: torch.Tensor) -> torch.Tensor:
        pass
    