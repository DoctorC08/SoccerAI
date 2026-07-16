import torch
from dataclasses import asdict
from src.buffers.base_buffer import BaseBuffer
from typing import List, Dict, Tuple, Any
import random
import numpy as np
from src.envs.transition import Transition

class TorchTensorBuffer(BaseBuffer):
    def __init__(self, 
                 max_size: int, 
                 device: torch.device, 
                 batch_size: int, 
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95
                ) -> None:
        super().__init__(max_size, device)

        # initialize empty memory
        self.temp_memory: Dict[str, torch.Tensor] = {}
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0

        self.is_buffer_finalized = False
        self.full_buffer = False

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.batch_size = batch_size

        assert max_size > 0, "max_size must be positive"
        assert batch_size is not None and batch_size > 0, "batch_size must be positive"

        if max_size % batch_size != 0:
            print("Warning: max_size is not a multiple of batch_size. Some data may be dropped during sampling.")
            print("max_size:", max_size, "batch_size:", batch_size)

    def get_data(self) -> Dict: 
        if not self.is_buffer_full and not self.is_buffer_finalized:
            print("Warning: Buffer is not full and/or buffer is not finalized")
        return self.buffer

    def add(self, data: dict) -> None:
        '''
        add a single experience to the buffer
        '''
        if self.index >= self.max_size:
            self.full_buffer = True
            return

        temp_dict = {}
        temp_dict['state'] = data.get('state')
        temp_dict['values'] = data.get('values')
        temp_dict['actions'] = data.get('actions')
        temp_dict['rewards'] = data.get('rewards')
        temp_dict['dones'] = data.get('dones')
        temp_dict['logits'] = data.get('logits')

        # Pre allocate memory
        keys = ['state', 'actions', 'rewards', 'dones', 'logits', 'values']
        if self.temp_memory == {}: 
            for key in keys:
                value = temp_dict[key]
                if value is None:
                    continue
                elif not isinstance(value, torch.Tensor): 
                    value = torch.as_tensor(value, device=self.device) 
                shape = (self.max_size, *value.shape)
                if key == 'dones':
                    dtype = torch.bool
                elif key == 'actions': 
                    dtype = torch.long 
                else:
                    dtype = torch.float32
                self.temp_memory[key] = torch.zeros(shape, dtype=dtype, device=self.device)
        
        for key in keys:
            if key not in self.temp_memory:
                continue
            value = temp_dict[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device)
            self.temp_memory[key][self.index] = value

        self.is_buffer_finalized = False
        self.index += 1

        if self.index >= self.max_size:
            self.full_buffer = True
        

    def is_buffer_full(self):
        return self.full_buffer 

    def finalize_buffer(self, next_state_value: Any = None) -> None:
        '''
        move temp_memory to device and into buffer
        '''
        self.buffer = self.temp_memory

        if next_state_value is None:
            num_envs = self.buffer['rewards'].shape[1]
            next_state_value = torch.zeros(num_envs, device=self.device)
        else:
            if not isinstance(next_state_value, torch.Tensor):
                next_state_value = torch.as_tensor(next_state_value, device=self.device)
            next_state_value = next_state_value.squeeze(-1) # shape (num_envs,)


        self.compute_returns_and_advantages(next_state_value=next_state_value)
        self.is_buffer_finalized = True

    def sample(self, batch_size: int = 0, clear_buffer: bool = True) -> List[Dict[str, torch.Tensor]]:
        if batch_size <= 0:
            batch_size = batch_size
        else: 
            batch_size = self.batch_size


        if not self.is_buffer_finalized:
            print("Finalizing buffer before sampling. Since no final state value passed in, using 0.0 as next_state_value.")
            self.finalize_buffer()

        if not self.is_buffer_full: 
            print("Warning: Sampling from a buffer that is not full.")

        # Flatten buffers across time and environment dimensions
        flat_buffer = {}
        for key, val in self.buffer.items():
            flat_buffer[key] = val.flatten(0, 1)

        buffer_size = len(flat_buffer['state'])
        indices = torch.randperm(buffer_size, device=self.device)

        minibatches = []
        for start_idx in range(0, buffer_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            minibatch = {key: value[batch_indices] for key, value in flat_buffer.items()}
            minibatches.append(minibatch)


        if clear_buffer:
            self.clear()

        return minibatches

    def compute_returns_and_advantages(self, next_state_value: float = 0.0) -> None: 
        '''
        Compute advantages and returns using GAE
        '''
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones'].float()

        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        
        T = rewards.size(0)

        next_value = next_state_value
        
        last_gae_lam = 0.0
        for step in reversed(range(T)):
            if step == T - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = advantages[step]
        
        returns = advantages + values
            
        self.buffer['advantage'] = advantages
        self.buffer['return'] = returns
    

    def clear(self): 
        self.temp_memory = {}
        self.buffer = {}

        self.index = 0
        self.full_buffer = False
        self.is_buffer_finalized = False

    
    