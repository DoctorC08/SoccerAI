from dataclasses import dataclass
import torch
from typing import List, Any, Optional
import numpy as np

@dataclass 
class Transition: 
    state: torch.Tensor
    next_state: torch.Tensor
    # length 1 for single-agent, n for MARL
    actions: List[Any] | np.ndarray
    rewards: List[float] | np.ndarray
    logits: torch.Tensor
    dones: bool | np.ndarray
    info: Any #TODO find datatype for this
    # value: torch.Tensor