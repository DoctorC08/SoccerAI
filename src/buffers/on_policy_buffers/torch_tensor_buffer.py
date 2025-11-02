import torch
from src.buffers.base_buffer import BaseBuffer
from typing import List, Dict, Tuple
import random
import numpy as np

class TorchTensorBuffer(BaseBuffer):
    def __init__(self, max_size: int, device: str, batch_size: int, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        super().__init__(max_size, device)

        # initialize empty memory
        self.temp_memory: Dict[str, list] = {}
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0

        self.is_buffer_finalized = False
        self.is_buffer_full = False

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.batch_size = batch_size

        assert max_size > 0, "max_size must be positive"
        assert batch_size is not None and batch_size > 0, "batch_size must be positive"

        if max_size % batch_size != 0:
            print("Warning: max_size is not a multiple of batch_size. Some data may be dropped during sampling.")
            print("max_size:", max_size, "batch_size:", batch_size)

    def add(self, data: dict) -> None:
        '''
        add a single experience to the buffer
        '''
        if self.index >= self.max_size:
            self.is_buffer_full = True
            return
        keys = ['state', 'action', 'reward', 'done', 'log_prob', 'value']
        if not self.temp_memory:
            self.temp_memory = {key: [] for key in keys}

        for key in data:
            if key in keys: 
                # if key == 'log_prob' or key == 'value':
                #     self.temp_memory[key].append(data[key].item())
                # else:
                self.temp_memory[key].append(data[key])
        
        self.is_buffer_finalized = False

        self.index += 1
        

    def is_buffer_full(self):
        return self.is_buffer_full 

    def finalize_buffer(self, next_state_value: float = 0.0) -> None:
        '''
        Convert temp_memory lists to tensors and store in buffer
        '''
        dtype_map = {'action': torch.long, 'done': torch.bool, 
                     'state': torch.float32, 'reward': torch.float32, 
                     'log_prob': torch.float32, 'value': torch.float32}

        self.buffer = {}
        for key, value in self.temp_memory.items():
            if key in dtype_map:
                # print(key, value)
                if key == 'state':
                    self.buffer[key] = torch.stack([torch.as_tensor(v, dtype=dtype_map[key], device=self.device) for v in value])
                else:
                    self.buffer[key] = torch.tensor(value, dtype=dtype_map[key], device=self.device)
            else:
                # Default to float32 if key not recognized
                print(f"Warning: Key {key} not recognized, defaulting to float32")
                self.buffer[key] = torch.tensor(value, dtype=torch.float32, device=self.device)

        self.temp_memory = {}

        self.compute_returns_and_advantages(next_state_value=next_state_value)

        self.is_buffer_finalized = True

    def sample(self, batch_size: int = None, clear_buffer: bool = True) -> List[Dict[str, List]]:
        
        if batch_size is not None:
            batch_size = self.batch_size
        else: 
            batch_size = self.batch_size


        if not self.is_buffer_finalized:
            print("Finalizing buffer before sampling. Since no final state value passed in, using 0.0 as next_state_value.")
            self.finalize_buffer()

        if not self.is_buffer_full: 
            print("Warning: Sampling from a buffer that is not full.")

        buffer_size = len(self.buffer['state'])
        indicies = torch.randperm(buffer_size, device=self.device)

        minibatches = []
        for start_idx in range(0, buffer_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indicies[start_idx:end_idx]

            minibatch = {key: value[batch_indices] for key, value in self.buffer.items()}
            minibatches.append(minibatch)


        if clear_buffer:
            self.clear()

        return minibatches

    def compute_returns_and_advantages(self, next_state_value: float = 0.0) -> None: 
        '''
        Compute advantages and returns using GAE
        '''
        rewards = self.buffer['reward']
        values = self.buffer['value']
        dones = self.buffer['done'].float()

        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        
        T = rewards.size(0)

        next_value = next_state_value
        
        last_return = rewards[-1] + self.gamma * next_value * (1.0 - dones[-1])
        last_delta = last_return - values[-1]

        advantages[-1] = last_delta
        returns[-1] = last_return

        for step in reversed(range(T - 1)):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
            advantages[step] = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * advantages[step + 1]
            
            returns[step] = advantages[step] + values[step]
            
        self.buffer['advantage'] = advantages
        self.buffer['return'] = returns
    

    def clear(self): 
        self.temp_memory: Dict[str, list] = {}
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0
        self.is_buffer_full = False
        self.is_buffer_finalized = False

    
    