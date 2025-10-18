from abc import ABC, abstractmethod
import torch 
from typing import List, Dict

class BaseBuffer(ABC):
    def __init__(self, max_size: int, device: str) -> None:
        self.max_size = max_size
        self.device = device

    @abstractmethod
    def add(self, *args) -> None:
        '''
        add a single experience to the buffer
        '''
        pass

    @abstractmethod
    def compute_returns_and_advantages(self, batch_size: int):
        '''
        calculate GAE and advantages
        '''
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        '''
        divide buffer into batch_size len dicts and return as list
        '''
        pass

    def clear(self) -> None:
        '''
        clear the buffer
        '''
        pass