from abc import ABC, abstractmethod
import torch 
from typing import List, Dict

class BaseBuffer(ABC):
    def __init__(self, max_size: int, device: str) -> None:
        self.max_size = max_size
        self.device = device
        self.is_buffer_finalized = False

    @abstractmethod
    def add(self, data: dict) -> None:
        '''
        add a single experience to the buffer
        '''
        pass

    @abstractmethod
    def sample(self, batch_size: int, clear_buffer: bool = True) -> List[Dict[str, torch.Tensor]]:
        '''
        divide buffer into dicts with values len=batch_size and stacks dicts into list

        Args:
            batch_size: size of each sample
            clear_buffer: clears buffers after sampling
        '''
        pass

    def clear(self) -> None:
        '''
        clear the buffer
        '''
        pass