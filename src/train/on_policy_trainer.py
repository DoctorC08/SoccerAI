import numpy as np
import torch
import math

from src.train.base_trainer import BaseTrainer
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.envs.transition import Transition

from src.eval.base_eval import BaseEval

class onPolicyTrainer(BaseTrainer):
    def __init__(self, 
                 agent, 
                 buffer, 
                 env, 
                 logger_config, 
                 evaluator: BaseEval,
                 eval_freq = 0, 
                 model_save_freq = 1000, 
                 model_save_path = './src/trained_agents/', 
                 save_best_model = True, 
                 best_model_exp_moving_avg = 0.95, 
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
            log_env_info=log_env_info, 
            env_info_fn=env_info_fn, 
            render_evals=render_evals, 
            fps=fps
            )

        self.reset_metrics()

    
    def init_logger(self, logger):
        models = self.agents.get_models()
        loss_fns = self.agents.get_loss_fns()
        zipped_data = zip(models, loss_fns)

        # Track all models: gradients and parameters 
        for i, (model, loss_fn) in enumerate(zipped_data):
            logger.watch_model(model, criterion=loss_fn, idx=i)
        
        return logger


    def init_agents(self, agents): 
        # Initialize agent
        return agents
    
    def get_batch_size(self, agents):
        return agents.batch_size

    def get_device(self):
        return super().get_device()

    def init_buffers(self, buffers):
        return buffers
    
    def clear_buffers(self):
        self.buffers.clear()
    
    def init_env(self, env):
        return env
    
    def init_eval(self, evaluator):
        return evaluator(self.render_evals, self.env, self.get_action, self.get_metrics)
    
    def collect_transition(self, state) -> Transition:
        action, logits = self.get_action(state, is_training=True)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        transition = Transition(
            state=state, 
            next_state=torch.from_numpy(next_state, dtype=torch.float32, device=self.device),
            actions=action, 
            rewards=reward, 
            logits=logits, 
            done=done,
            info=info,
        )
        return transition

    def get_action(self, state, is_training):
        # Get action, return action, logits 
        action, logits = self.agents.select_action(state.to(self.agents.device), is_training=is_training)
        return action, logits

    def update_buffer(self, state, action, reward, done, logits, next_state) -> None: 

        # If on-policy add value estimates to buffer
        if isinstance(state, torch.Tensor):
            state_value = self.agents.find_value(state.to(self.agents.device))
        elif isinstance(state, np.ndarray):
            state_value = self.agents.find_value(torch.tensor(state, dtype=torch.float32).to(self.agents.device))
        else:
            raise TypeError(f"Unknown state type returned from env: {type(state)}")
        self.buffers.add({
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "log_prob": logits,
            "value": state_value.detach().cpu().item(),
        })
        if self.buffers.is_buffer_full:
            self.buffers.finalize_buffer(self.agents.find_value(next_state.to(self.agents.device)).detach().cpu().item())
            self.update()
            self.buffers.clear()

    
    def update_agents(self):
        cur_updates = 0
        # update agents and return update_metrics, number of updates
        # update self.n_updates 
        for epoch in range(self.agents.n_epochs):
            sample_data = self.buffers.sample(self.batch_size, clear_buffer=False)
            if isinstance(sample_data, list) and isinstance(self.buffers, TorchTensorBuffer):
                # assuming if sample_data is single list then each item will hold dict with values lists of experiences
                for i, experience in enumerate(sample_data):
                    update_metrics = self.agents.update(states=experience['state'], 
                                                        returns=experience['return'], 
                                                        advantages=experience['advantage'], 
                                                        actions=experience['action'],
                                                        identifier=i)
                    self.n_updates += 1
                    cur_updates += 1
            else:
                raise TypeError("Unexpected sample data type during off-policy update or unexpected buffer type")
        
        # Clear buffer after epoch updates
        self.buffers.clear()
        
        return update_metrics, cur_updates



    def get_metrics(self, logits, logger_dir): 
        # update any training metrics, return a dict of metrics to be logged or used
        if self.ep_train_entropy is None:
            self.ep_train_entropy = 0

        # Find "certainty" by using entropy H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)
        self.ep_train_entropy += (-logits.exp() * logits).sum().item()

        return {
            f"{logger_dir}ep_train_entropy": self.ep_train_entropy,
        }

    def reset_metrics(self):
        self.ep_train_entropy = 0


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

        