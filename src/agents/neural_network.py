import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(torch.nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: list[int], 
                 output_size: int, 
                 final_activation: torch.nn.Module = None,
                ) -> None:
        '''
           Keep final_activation to None unless you want a final activation, 
           but generally handle this later so forward method can return logits. 
           
           Using ReLU activation between layers. 
        '''
        super().__init__()

        if isinstance(input_size, tuple):
            # Use numpy's prod to handle any (D,) or (H, W, C) input shape
            print(f"Warning: input_size is a tuple: {input_size}")
            input_size = int(np.prod(input_size)) 
            print(f"Warning: had to convert input_size to flat integer: {input_size}")
            
        # Ensure output_size is a flat integer
        if isinstance(output_size, tuple):
            print(f"Warning: output_size is a tuple: {output_size}")
            output_size = int(np.prod(output_size))
            print(f"Warning: had to convert output_size to flat integer: {output_size}")

        layers = []
        try: 
            cur_input_size = input_size
            if hidden_sizes:
                layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[0]))
                layers.append(nn.ReLU())
                cur_input_size = hidden_sizes[0]
                
                for i in range(len(hidden_sizes) - 1):
                    layers.append(nn.Linear(in_features=cur_input_size, out_features=hidden_sizes[i+1]))
                    layers.append(nn.ReLU())
                    cur_input_size = hidden_sizes[i+1]
        except Exception as e:
            print(f"Error in constructing models: {e}")
            print(f"input_size: {input_size} \n hidden_sizes: {hidden_sizes} \n output_size: {output_size}")
            raise e

        layers.append(nn.Linear(in_features=cur_input_size, out_features=output_size))
        if final_activation: 
            print("Warning: final activation detected")
            layers.append(final_activation)
            
        self.model = nn.Sequential(*layers)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.model(input_data)