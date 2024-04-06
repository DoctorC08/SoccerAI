import numpy as np
import torch

lista = [torch.tensor([1, 2]), torch.tensor([3, 4])]

listc = torch.stack(lista)

print(listc)
