from abc import ABC, abstractmethod
import torch 
from typing import List, Dict, Any

from src.envs.transition import Transition

class BaseBuffer(ABC):
    def __init__(self, max_size: int, device: torch.device) -> None:
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
    def is_buffer_full(self) -> bool:
        '''
        Returns true if buffer is full
        '''
        pass 

    @abstractmethod
    def finalize_buffer(self, next_state_value) -> None:
        '''
        Finalize the buffer - move to final device, compute any needed values
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

    @abstractmethod
    def get_data(self) -> Dict: 
        '''
        return all the data as a single dict
        '''
        pass