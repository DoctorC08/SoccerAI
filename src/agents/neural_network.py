import torch
import torch.nn as nn


class NeuralNetwork(torch.nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: list[int], 
                 output_size: int, 
                 final_activation: torch.nn.Module = None,
                ) -> None:
        '''
           Keep final_activation to None unless you want a final activation, 
           but generally handle this later so forward method can return logits
        '''
        super().__init__()

        layers = []
        
        cur_input_size = input_size
        if hidden_sizes:
            layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[0]))
            layers.append(nn.ReLU())
            cur_input_size = hidden_sizes[0]
            
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(in_features=cur_input_size, out_features=hidden_sizes[i+1]))
                layers.append(nn.ReLU())
                cur_input_size = hidden_sizes[i+1]

        layers.append(nn.Linear(in_features=cur_input_size, out_features=output_size))
        if final_activation: 
            print("Warning: final activation detected")
            layers.append(final_activation)
            
        self.model = nn.Sequential(*layers)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.model(input_data)