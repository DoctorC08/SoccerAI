from dataclasses import dataclass
import torch
from typing import List, Any, Optional

@dataclass 
class Transition: 
    state: torch.Tensor
    next_state: torch.Tensor
    # length 1 for single-agent, n for MARL
    actions: List[Any] 
    rewards: List[float] 
    logits: List[Optional[torch.Tensor]]
    done: bool
    info: Any #TODO find datatype for this