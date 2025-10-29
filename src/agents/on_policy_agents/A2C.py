from gymnasium import spaces
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .policy_agent import PolicyAgent

class A2CAgent(PolicyAgent):
    def __init__(self,
                 state_size: spaces.Space, 
                 action_size: spaces.Space,
                 policy_network: nn.Module, 
                 critic_network: nn.Module,               
                 learning_rate: float = 1e-4,
                 grad_clip: float = 1,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 optimizer: torch.optim.Optimizer = None,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps"),
                 **kwargs
                ) -> None:
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            policy_network=policy_network,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            entropy_coef=entropy_coef,
            device = device,
            **kwargs
        )
        self.critic_network = critic_network
        self.value_loss_coef = value_loss_coef

        # Assuming same optimizer for both actor and critic
        self.critic_optimizer = optimizer if optimizer else torch.optim.Adam(
            list(self.policy_network.parameters()) + list(self.critic_network.parameters()), 
            lr=self.learning_rate
        )
        self.actor_optimizer = optimizer if optimizer else torch.optim.Adam(
            list(self.policy_network.parameters()) + list(self.critic_network.parameters()), 
            lr=self.learning_rate
        )
        
    def _setup_model(self) -> None:
        self.policy_network.to(self.device)
        self.critic_network.to(self.device)

    def select_action(self, state: torch.Tensor, is_training: bool = True) -> Tuple[int, torch.Tensor]:
        state = state.to(self.device)
        logits = self.policy_network(state)
        if is_training:
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample() 
            return action.item(), logits
        else:
            action = torch.argmax(logits, dim=-1).item()
            return action, logits

    def update(self, states, returns, advantages, log_prob, identifier=None) -> Dict[str, float]:
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        predictions = self.critic_network(states).squeeze(-1)

        critic_loss = nn.MSELoss()(predictions, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        actor_loss = -torch.mean(advantages * log_prob)
        entropy_loss = -torch.mean(torch.exp(log_prob) * log_prob)
        actor_loss += self.entropy_coef * entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.grad_clip)
        self.actor_optimizer.step()

        return {
            f"loss/actor_loss_{identifier}": actor_loss.item(),
            f"loss/critic_loss_{identifier}": critic_loss.item(),
            f"loss/entropy_loss_{identifier}": entropy_loss.item(),
        }



    

