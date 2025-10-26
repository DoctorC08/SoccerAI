import torch
from src.buffers.base_buffer import BaseBuffer
from typing import List, Dict
import random
import numpy as np

class TorchTensorBuffer(BaseBuffer):
    def __init__(self, max_size: int, device: str) -> None:
        super().__init__(max_size, device)

        # initialize empty memory
        self.temp_memory: Dict[str, list] = {}
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0

        self.is_buffer_finalized = False

    def add(self, data: dict) -> None:
        '''
        add a single experience to the buffer
        '''
        keys = ['state', 'action', 'reward', 'done', 'log_prob']
        if not self.temp_memory:
            self.temp_memory = {key: [] for key in keys}
        
        # TODO: Ensure we have all the keys
        # if len(data) != len(keys):
        #     raise ValueError(f"Expected {len(keys)} arguments, got {len(args)}")

        for key in data:
            if key in keys: 
                self.temp_memory[key].append(data[key])
        
        self.is_buffer_finalized = False
        
        self.index += 1
        # TODO: Should I be looping around instead of just going over buffer size? 
        # Like this? If so should assert that self.max_length % batch_size == 0 ?
        # self.index = (self.index + 1) % self.max_length
        # self.n = min(self.n + 1, self.max_length)
        
    
    def is_buffer_full(self):
        return self.index >= self.max_size

    def finalize_buffer(self) -> None:
        '''
        Convert temp_memory lists to tensors and store in buffer
        '''
        self.buffer = {key: torch.tensor(value, dtype=torch.float32 if key != 'action' else torch.long, device=self.device) 
                       for key, value in self.temp_memory.items()}
        
        for key, value in self.temp_memory.items():
            if key == 'action' or key == 'done': 
                self.buffer[key] = torch.tensor(value, dtype=torch.long, device=self.device)
            else: 
                self.buffer[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        self.temp_memory = {}

        self.is_buffer_finalized = True

    def sample(self, batch_size: int, clear_buffer: bool = True) -> List[Dict[str, List]]:

        if not self.is_buffer_finalized:
            self.finalize_buffer()

        # Generate random samples from buffer
        samples = []
        keys = ['state', 'action', 'reward', 'done', 'log_prob', 'value', 'advantage', 'return']

        assert self.buffer['advantages'] is not None or \
            self.buffer['returns'] is not None, \
                "Calculations for advantageess and returns need to happen before sample call"

        # Find random indicies
        indicies = [i for i in range(self.index)]
        indicies = np.random.choice(indicies, replace=False, size=self.index)

        sample_index = 0
        
        for _ in range(self.index // batch_size):
            sample_buffer = {i : [] for i in keys}

            for _ in range(batch_size):
                
                for key in keys:
                    sample_buffer[key].append(self.buffer[key][indicies[sample_index]])

                sample_index += 1
            
            samples.append(sample_buffer)
        
        if not self.is_buffer_full(): 
            print("Warning: Buffer not full when sampled")
            print(f"Only at index {self.index} out of maximum size of {self.max_size}")
        
        if self.index % batch_size != 0: 
            print(f"Warning: {self.index % batch_size} data will be missing")
        
        if clear_buffer:
            self.clear()

        return samples


    def clear(self): 
        self.temp_memory: Dict[str, list] = {}
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0

    