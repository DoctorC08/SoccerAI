from gymnasium import spaces
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.agents.on_policy_agents.policy_agent import PolicyAgent

class A2CAgent(PolicyAgent):
    def __init__(self,
                 state_size: spaces.Space | int, 
                 action_size: spaces.Space | int,
                 policy_network: nn.Module, 
                 critic_network: nn.Module,     
                 optimizer: str,
                 learning_rate: float = 1e-4,
                 batch_size: int = 256,
                 grad_clip: float = 1,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps"),
                 **kwargs
                ) -> None:
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            entropy_coef=entropy_coef,
            device = device,
            batch_size=batch_size,
            **kwargs
        )
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.value_loss_coef = value_loss_coef

        # Assuming same optimizer for both actor and critic
        if optimizer == "ADAM" or optimizer is None: 
            self.critic_optimizer = torch.optim.Adam(
                list(self.policy_network.parameters()) + list(self.critic_network.parameters()), 
                lr=self.learning_rate
            )
            self.actor_optimizer = torch.optim.Adam(
                list(self.policy_network.parameters()) + list(self.critic_network.parameters()), 
                lr=self.learning_rate
            )
        else:
            raise NotImplementedError(f"Unexpected optimizer type: {optimizer}")

        self._setup_model()
        
    def _setup_model(self) -> None:
        if self.policy_network and self.critic_network:
            self.policy_network.to(self.device)
            self.critic_network.to(self.device)
            print("Policy and Critic networks moved to device:", self.device)
        else:
            print("Policy or Critic network not provided!")
            print("Policy Network:", self.policy_network)
            print("Critic Network:", self.critic_network)

    def select_action(self, state: torch.Tensor, is_training: bool = True) -> Tuple[int, torch.Tensor]:
        state = state.to(self.device, dtype=torch.float32)
        if state.ndim == 0:
            state = state.unsqueeze(0)
        logits = self.policy_network(state)
        if logits.ndim > 1 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        action_dist = torch.distributions.Categorical(logits=logits)
        if is_training:
            action = action_dist.sample() 
            
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob 
        else:
            action = torch.argmax(logits, dim=-1)
            action_item = action.item() 
            
            log_prob = action_dist.log_prob(action)
            
            return action_item, log_prob

    def update(self, states, returns, advantages, actions, identifier=None) -> Dict[str, float]:
        states = states.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        actions = actions.to(self.device)

        predictions = self.critic_network(states).squeeze(-1)

        critic_loss = nn.MSELoss()(predictions, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        logits = self.policy_network(states)
        action_dist = torch.distributions.Categorical(logits=logits)
        log_probs = action_dist.log_prob(actions)

        actor_loss = -torch.mean(advantages.detach() * log_probs)
        entropy_loss = action_dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.grad_clip)
        self.actor_optimizer.step()

        return {
            f"update/actor_loss_{identifier}": actor_loss.item(),
            f"update/critic_loss_{identifier}": critic_loss.item(),
            f"update/entropy_loss_{identifier}": entropy_loss.item(),
        }
    
    def find_value(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        value = self.critic_network(state)
        return value
    
    def save_model(self, path: str) -> None:
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'critic_network_state_dict': self.critic_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        # print(f"Model saved to {path}") 
    
    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")
        self.policy_network.to(self.device)
        self.critic_network.to(self.device)
        print(f"Models automatically moved to device: {self.device}")

    def get_models(self) -> List[nn.Module]:
        return [self.policy_network, self.critic_network]
    
    def get_loss_fns(self):
        return [None, nn.MSELoss()]



    

