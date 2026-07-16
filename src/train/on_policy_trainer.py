import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import math
from typing import Tuple, Any

from src.train.base_trainer import BaseTrainer
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.envs.transition import Transition

from src.eval.base_eval import BaseEval

class onPolicyTrainer(BaseTrainer):
    def __init__(self, 
                 agent: PolicyAgent, 
                 buffer, 
                 env, 
                 logger_config, 
                 evaluator: BaseEval,
                 eval_freq = 0, 
                 model_save_freq = 1000, 
                 model_save_path = './src/trained_agents/', 
                 save_best_model = True, 
                 best_model_exp_moving_avg = 0.95, 
                 n_envs: int = 1, 
                 log_env_info = False, 
                 env_info_fn=None, 
                 render_evals=True, 
                 fps = 5):
        super().__init__(
            agent=agent, 
            buffer=buffer, 
            env=env, 
            logger_config=logger_config, 
            evaluator=evaluator, 
            eval_freq=eval_freq, 
            model_save_freq=model_save_freq, 
            model_save_path=model_save_path, 
            save_best_model=save_best_model, 
            best_model_exp_moving_avg=best_model_exp_moving_avg, 
            n_envs = n_envs,
            log_env_info=log_env_info, 
            env_info_fn=env_info_fn, 
            render_evals=render_evals, 
            fps=fps
            )

        self.reset_metrics() 
    
    def init_agents(self, agents: PolicyAgent) -> PolicyAgent: 
        # Initialize agent
        return agents

    def init_eval(self, evaluator):
        return evaluator(self.render_evals, self.env, self.get_action, self.get_metrics)
    
    def collect_transition(self, state) -> Transition:
        action, logits = self.get_action(state, is_training=True)
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = np.logical_or(terminated, truncated)
        transition = Transition(
            state=state, 
            next_state=torch.as_tensor(next_state, dtype=torch.float32, device=self.device),
            actions=action, 
            rewards=reward, 
            logits=logits, 
            dones=done,
            info=info,
        )
        return transition

    def get_action(self, state, is_training) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get action, return action, logits 
        actions, logits = self.agents.select_action(state.to(self.agents.device), is_training=is_training)
        return actions, logits

    def update_buffer(self, transition) -> None: 
        # If on-policy add value estimates to buffer
        state_value = self.agents.find_value(transition.state.to(self.agents.device)).detach().squeeze(-1)


        self.buffers.add({
            "state": transition.state,
            "actions": transition.actions,
            "rewards": transition.rewards,
            "dones": transition.dones,
            "logits": transition.logits,
            "values": state_value,
        })

        if self.buffers.is_buffer_full:
            self.buffers.finalize_buffer(self.agents.find_value(transition.next_state.to(self.agents.device)).detach())
            # self.update()
            # self.buffers.clear()

    
    def update_agents(self) -> Tuple[Any, int]:
        # update agents and return update_metrics, number of updates
        cur_updates = 0
        update_metrics = {}

        buffer_data = self.buffers.get_data()
        
        states = buffer_data['state'].flatten(0, 1)
        actions = buffer_data['actions'].flatten(0, 1)
        advantages = buffer_data['advantage'].flatten(0, 1)
        returns = buffer_data['return'].flatten(0, 1)

        dataset = TensorDataset(states, actions, advantages, returns)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0
        )

        # update self.n_updates 
        for epoch in range(self.agents.n_epochs):
            for batch_idx, (b_states, b_actions, b_advantages, b_returns) in enumerate(dataloader):
                # Calling specific on policy update method
                metrics = self.agents.update(
                    states=b_states, 
                    returns=b_returns, 
                    advantages=b_advantages, 
                    actions=b_actions,
                    identifier=f"epoch_{epoch}_batch_{batch_idx}"
                )
                update_metrics.update(metrics)
                self.n_updates += 1
                cur_updates += 1

        self.buffers.clear()
        
        return update_metrics, cur_updates



    def get_metrics(self, logits, logger_dir): 
        # update any training metrics, return a dict of metrics to be logged or used
        if self.ep_train_entropy is None:
            self.ep_train_entropy = [0 for _ in range(self.n_envs)]

        # Find "certainty" by using entropy H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)
        self.ep_train_entropy += (-logits.exp() * logits)

        return {
            f"{logger_dir}ep_train_entropy": self.ep_train_entropy,
        }

    def reset_ind_metrics(self, i):
        self.ep_train_entropy[i] = 0

    def reset_metrics(self):
        self.ep_train_entropy = torch.zeros(self.n_envs, device=self.device)


    def save_agents(self, path_name): 
        self.agents.save_model(path_name)

    def skip_update(self):
        # Always skip update in normal training loop and only update when buffer is full
        if self.buffers.is_buffer_full: 
            return False
        else: 
            return True

    def validate_params(self, agents, buffers, logger):
        assert isinstance(agents, PolicyAgent)
        assert agents.batch_size > 0, "Batch size must be positive for on-policy agents"
        assert self.eval_freq > 0, "Evaluation frequency must be positive for on-policy agents"
        assert agents.batch_size <= self.buffers.max_size, f"Batch size must be less than or equal to buffer size for on-policy agents. Current buffer size: {self.buffers.max_size}, batch size: {self.batch_size}"
        assert buffers.max_size % self.batch_size == 0, "Buffer size must be multiple of batch size for on-policy agents"
        if isinstance(self.agents, A2CAgent):
            assert agents.n_epochs == 1, "Number of epochs must be 1 for A2C agents"
        else:
            assert agents.n_epochs > 0, "Number of epochs must be positive for on-policy agents"

        