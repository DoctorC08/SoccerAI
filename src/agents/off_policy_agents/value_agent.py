from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from src.agents.base_agent import BaseAgent

@dataclass
class ValueAgent(BaseAgent, ABC):
    
    target_network: torch.nn.Module
    
    epsilon: float = .5
    epsilon_decay: float = 0.95
    epsilon_min: float = 0.01

    batch_size: int = 64
    memory_size: int = 100_000
    update_target_freq: int = 100

    n_update_steps: int = 1
    model_update_freq: int = 1

    

    