import torch
from src.buffers.base_buffer import BaseBuffer
from typing import Dict

class TorchDataLoaderDeque(BaseBuffer):
    def __init__(self, 
                 max_size: int, 
                 device: torch.device, 
                ) -> None:
        super().__init__(max_size, device)

        # initialize empty memory
        self.buffer: Dict[str, torch.Tensor] = {}

        self.index = 0

        self.full_buffer = False

    def get_data(self) -> Dict: 
        if not self.is_buffer_full: 
            print("Warning: Buffer is not yet full")
        return self.buffer

    def add(self, data: dict) -> None:
        '''
        add a single experience to the buffer
        '''

        temp_dict = {}
        temp_dict['state'] = data.get('state')
        temp_dict['values'] = data.get('value')
        temp_dict['actions'] = data.get('actions')
        temp_dict['rewards'] = data.get('rewards')
        temp_dict['dones'] = data.get('dones')

        # Pre allocate memory
        keys = ['state', 'actions', 'rewards', 'dones', 'logits', 'values']
        if self.buffer == {}: 
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
                self.buffer[key] = torch.zeros(shape, dtype=dtype, device=self.device)
        
        for key in keys:
            if key not in self.buffer:
                continue
            self.buffer[key][self.index] = temp_dict[key]

        self.index += 1

        if self.index >= self.max_size:
            self.full_buffer = True
            self.index = 0
        

    def is_buffer_full(self):
        return self.full_buffer 

    def clear(self): 
        self.buffer = {}

        self.index = 0
        self.full_buffer = False

    
    