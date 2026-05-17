from dataclasses import dataclass, field
from typing import Any
from src.utils.config import AgentConfig, BufferConfig
from src.agents import base_agent
from src.buffers import base_buffer

# Dict[List[BaseAgent, BaseBuffer, bool, int, int, int]] | Dict[List[BaseAgent, BaseBuffer, bool, int, int, int, int]]
#                 On policy agent list config: agent, buffer, is_on_policy, batch_size, n_epochs, team
#                 Off policy agent list config: agent, buffer, is_on_policy, batch_size, model_update_freq, n_update_steps, team
#                 if team = 0 then no associated team and will just take 0th reward

@dataclass
class MARLAgentConfig:
    is_on_policy: bool 
    agent: base_agent
    buffer: base_buffer
    batch_size: int
    n_epochs: int
    team: int
    model_update_freq: int
    n_update_steps: int
    