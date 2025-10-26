from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from ..base_agent import BaseAgent

@dataclass
class PolicyAgent(BaseAgent, ABC):
    entropy_coef: float

    
    