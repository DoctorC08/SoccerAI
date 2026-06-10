import math
from abc import ABC, abstractmethod
import torch
import numpy as np
import time
import wandb
from tqdm import tqdm
import os

from src.buffers.base_buffer import BaseBuffer
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.utils.config import LoggerConfig

from src.loggers.wandb_logger import WandBLogger
from src.envs.mod_base_gym_env import modBaseGymEnv
from src.envs.transition import Transition

from src.agents.base_agent import BaseAgent
from src.agents.off_policy_agents.value_agent import ValueAgent
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent

from src.eval.base_eval import BaseEval
from src.eval.single_agent_eval

class BaseTrainer(ABC):
    def __init__(self, 
                agent: BaseAgent, 
                buffer: BaseBuffer,
                env: modBaseGymEnv, # Custom modified base gym
                logger_config: LoggerConfig, 
                evaluator: BaseEval,
                eval_freq: int = 100,
                model_save_freq: int = 1000,
                model_save_path: str = './src/trained_agents/',
                save_best_model: bool = True,
                best_model_exp_moving_avg: float = 0.95,
                log_env_info: bool = False,
                env_info_fn = None,
                render_evals = True,
                fps: int = 5,
            ):
        
        # Initialize agents, buffer, and logger
        self.name = logger_config.logger
        self.agents = self.init_agents(agent)
        self.buffers = self.init_buffers(buffer)
        
        if logger_config.logger == "WandBLogger":
            self.logger = WandBLogger(
                project = logger_config.project, 
                name = logger_config.name, 
                config = logger_config.wandb_config, 
                reinit = logger_config.reinit,
                sweep = logger_config.sweep,
            )
        else: 
            raise LookupError(f"Unknown logger inputed: {logger_config.logger}")
        self.logger = self.init_logger(self.logger)
        self.logger_save_freq = logger_config.logger_save_freq
        self.logger_train_update_speed = 0

        self.batch_size = self.get_batch_size(self.agents)


        # Initialize env
        self.env = self.init_env(env)
        self._state = None
        self.ep_rews = 0.0 
        self.ind_ep_rews = []
        self.log_env_info = log_env_info
        self.env_info_fn = env_info_fn
        self.env_info = None

        self.render_evals = render_evals
        self.fps = fps
        if self.render_evals:
            print(f"Saving eval renderings. fps set to {self.fps}")

        if log_env_info and env_info_fn is None:
            raise TypeError("log_env_info defined, but env_info_fn not defined. ")
        if log_env_info:
            print("Warning: logging environment info. " \
            "Assuming info is being passed back as appropriate type to be passed through env_info_fn and logged")
        
        # Create vars for saving best model
        self.eval_freq = eval_freq
        self.save_best_model = save_best_model
        if self.save_best_model:
            self.cur_score = 0.0
            self.best_model_exp_moving_avg = best_model_exp_moving_avg
            self.save_threshold = -math.inf
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        # Initialize logging vars
        self.n_train_step = 0
        self.n_updates = 0
        self.n_eps = 0

        self.device = self.get_device()

        # Initialize evalutaor
        self.evalutaor = self.init_evaluator(self, evaluator)

        # Validate params
        self.validate_params(self.agents, self.buffers, self.logger)

    @abstractmethod
    def init_logger(self, logger):
        pass

    @abstractmethod 
    def init_agents(self, agents): 
        # Likely will either be a single BaseAgent or List[BaseAgent]
        pass

    @abstractmethod
    def get_batch_size(self, agents):
        # get a batch size to log: update_frames_per_sec": cur_updates * self.batch_size / delta_t,
        pass

    @abstractmethod
    def get_device(self) -> torch.DeviceObjType:
        '''
        get device
        '''
        return self.agents.device

    @abstractmethod 
    def init_buffers(self, buffers):
        # Initialize buffers
        pass 
    
    @abstractmethod
    def init_env(self, env):
        # Initialize env
        pass 

    @abstractmethod
    def init_eval(self, evaluator):
        # Initialize eval
        pass 

    @abstractmethod
    def collect_transition(self, state) -> Transition:
        '''
        collect a full transition
        '''
        pass

    @abstractmethod
    def get_action(self, state, is_training):
        '''
        Get action, return action, logits (if logits not used then return None)
        '''
        pass

    @abstractmethod
    def update_buffer(self, transition: Transition) -> None: 
        '''
        Add new data into buffer and call updates/clear if needed

        Extra variables should be None if not used
        Next state is used when finalizing buffer for on policy methods
        if method is on policy also calculate values for each state
           in addition, if it's full finalize buffer, update, and then clear
        '''
        pass

    @abstractmethod
    def clear_buffers(self): 
        '''
        Clear buffers
        '''
        pass


    @abstractmethod 
    def update_agents(self):
        '''
        update agents and return update_metrics, number of updates
        pdate self.n_updates 
        '''
        pass

    @abstractmethod
    def get_metrics(self, logits, logger_dir): 
        '''
        update  training metrics, return a dict of metrics to be logged or used
        '''
        # ex. self.ep_train_certainty += self.calc_certainty(logits)
        pass

    @abstractmethod
    def reset_metrics(self):
        '''
        reset any training metrics at the end of an episode
        '''
        pass

    @abstractmethod
    def save_agents(self, path_name): 
        '''
        save agents 
        '''
        pass

    @abstractmethod
    def skip_update(self):
        '''
        check if update needs to be called
        for off policy agents should only be not skipping if self.t % self.model_update_freq == 0 and self.t > 0
        '''
        pass

    @abstractmethod 
    def validate_params(self, agents, buffers, logger):
        '''
        Validata parameter choices passed in
        '''
        pass 
        


    def train(self, n_steps: int, num_post_eval_runs: int = 0) -> None: 
        self.t = 0
        for t in tqdm(range(n_steps)): 
            if t % self.eval_freq == 0: 
                self.evaluator.eval()
            
            self.run_step()
            self.update()

            self.t += 1

        #TODO: move eval into it's own class
        if num_post_eval_runs: 
            eval_metrics = {}
            for _ in range(num_post_eval_runs):
                cur_metrics = self.evaluator.eval(log=False)
                for key, value in cur_metrics.items(): 
                    eval_metrics[key] = eval_metrics.get(key, 0) + value

            # average values
            for key in eval_metrics.keys():
                eval_metrics[key] /= num_post_eval_runs
            print("Eval metrics:", eval_metrics)
        
            self.logger.log(eval_metrics)

        self.cleanup()

        print(f"Stopping training as timesteps = {self.n_train_step}")
        print(f"and max training steps {n_steps}")
        print(f"{t} total steps ran")
        print(f"{self.n_train_step} training steps ran")
        print(f"{self.n_eps} episodes ran")
        print(f"{self.n_updates} model updates ran")

    def run_step(self):
        if self.log_env_info:
            self.env_info = []
        
        if self.ep_rews is None: 
            self.ep_rews = 0
        if self._state is None: 
            self._state, info = self.env.reset()
            # send state to a tensor
            if not isinstance(self._state, torch.Tensor):
                self._state = torch.as_tensor(self._state, dtype=torch.float32, device=self.device)
        if self.env.render_mode is not None: 
            self.env.change_render_mode(None)


        #TODO: Define collect_transition and transition dataclass to allow for MARL envs/trainers
        transition = self.collect_transition(self, self._state)

        # find state value if on policy
        # update buffer
        self.update_buffer(self._state, transition=transition)

        self._state = transition.next_state

        if self.ind_ep_rews is None:
            self.ind_ep_rews = [i for i in transition.rewards]

        self.ep_rews += sum(transition.rewards)

        self.ind_ep_rews += transition.rewards

        if self.log_env_info:
            self.env_info.append(info)

        metrics = self.get_metrics(transition.logits, logger_dir="train/")

        self.n_train_step += 1

        # Log episodic values if done
        if transition.done:
            self.n_eps += 1
            if self.n_eps % self.logger_save_freq == 0: 
                cur_time = time.time()
                if self.log_env_info:
                    self.logger.log({
                        "train/ep_rewards": self.ep_rews, 
                        "train/done": transition.done, 
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                        "train/env_info": self.env_info_fn(self.env_info), 
                        } | metrics, self.n_eps
                    )
                    self.env_info = []
                else: 
                    self.logger.log({
                        "train/ep_rewards": self.ep_rews, 
                        "train/done": transition.done, 
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                    } | metrics, 
                    self.n_eps
                    )
                
                self.logger_train_update_speed = cur_time

            # reset values
            self.ep_rews = 0
            self._state = None # Force env to reset
            self.reset_metrics()


    
    def update(self) -> bool:
        '''
        update agents and log time
        returns bool if agents are updated
        '''
        if self.skip_update(): 
            return False

        _start_update_time = time.time()

        # sample buffer
        # inumerate through data and update
        update_metrics, cur_updates = self.update_agents()


        end_update_time = time.time()
        delta_t = end_update_time - _start_update_time
        update_metric_speeds = {
            "update/updates_per_sec": cur_updates / delta_t, 
            "update/update_frames_per_sec": cur_updates * self.batch_size / delta_t,
            "update/num_updates": cur_updates
        }
        self.logger.log(
            update_metrics | update_metric_speeds, 
            self.n_eps
        )
        return False

    def cleanup(self):
        self.logger.close()
        self.clear_buffers()
        self.env.close()


    