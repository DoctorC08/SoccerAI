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

from src.agents.base_agent import BaseAgent
from src.agents.off_policy_agents.value_agent import ValueAgent
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent

class BaseTrainer(ABC):
    def __init__(self, 
                agent: BaseAgent, 
                buffer: BaseBuffer,
                env: modBaseGymEnv, # Custom modified base gym
                logger_config: LoggerConfig, 
                eval_freq: int = 0,
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
    def init_buffers(self, buffers):
        # Initialize buffers
        pass 
    
    @abstractmethod
    def init_env(self, env):
        # Initialize env
        pass 

    @abstractmethod
    def get_action(self, state, is_training):
        '''
        Get action, return action, logits (if logits not used then return None)
        '''
        pass

    @abstractmethod
    def update_buffer(self, state, action, reward, done, logits, next_state) -> None: 
        '''
        Add new data into buffer and call updates/clear if needed

        Extra variables should be None if not used
        Next state is used when finalizing buffer for on policy methods
        if method is on policy also calculate values for each state
           in addition, if it's full finalize buffer, update, and then clear
        '''
        pass

    @abstractmethod
    def sample_data(self, clear_buffer=True) -> None:
        '''
        Sample data from the buffers
        Used in update step
        '''
        pass

    @abstractmethod 
    def update_agents(self):
        # update agents and return update_metrics, number of updates
        # update self.n_updates 
        
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
        # self.is_off_policy = isinstance(self.agents, ValueAgent) 
        # self.is_on_policy = isinstance(self.agents, PolicyAgent)

        # if self.is_on_policy:
        #     assert self.batch_size > 0, "Batch size must be positive for on-policy agents"
        #     assert self.eval_freq > 0, "Evaluation frequency must be positive for on-policy agents"
        #     assert self.batch_size <= self.buffers.max_size, f"Batch size must be less than or equal to buffer size for on-policy agents. Current buffer size: {self.buffer.max_size}, batch size: {self.batch_size}"
        #     assert self.buffers.max_size % self.batch_size == 0, "Buffer size must be multiple of batch size for on-policy agents"
        #     if isinstance(self.agents, A2CAgent):
        #         assert agents.n_epochs == 1, "Number of epochs must be 1 for A2C agents"
        #     else:
        #         assert agents.n_epochs > 0, "Number of epochs must be positive for on-policy agents"

        # elif self.is_off_policy:
        #     assert agents.n_update_steps > 0, "Number of update steps must be positive for off-policy agents"
        #     assert agents.model_update_freq > 0, "Model update frequency must be positive for off-policy agents"
        # else: 
        #     raise TypeError("Agent must be either on-policy or off-policy type. Unknown type used")
        pass 
        


    def train(self, n_steps: int, num_post_eval_runs: int = 0) -> None: 
        self.t = 0
        for t in tqdm(range(n_steps)): 
            if t % self.eval_freq == 0: 
                self.eval()
            
            self.run_step()
            self.update()

            self.t += 1

        if num_post_eval_runs: 
            eval_metrics = {}
            for _ in range(num_post_eval_runs):
                cur_metrics = self.eval(log=False, return_eval_metrics=True)
                for key, value in cur_metrics.items(): #TODO: .items vs .values tinme
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
        print(f"{self.n_update_steps} update steps ran")

    def run_step(self):
        if self.log_env_info:
            self.env_info = []
        
        if self.ep_rews is None: 
            self.ep_rews = 0
        if self._state is None: 
            self._state, info = self.env.reset()
        if self.env.render_mode is not None: 
            self.env.change_render_mode(None)

        action, logits = self.get_action(self._state, is_training=True)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # find state value if on policy
        # update buffer
        self.update_buffer(self._state, action, reward, done, logits, next_state)

        self._state = next_state

        self.ep_rews += reward
        if self.log_env_info:
            self.env_info.append(info)

        metrics = self.get_metrics(logits, logger_dir="train/")

        self.n_train_step += 1

        # Log episodic values if done
        if done:
            self.n_eps += 1
            if self.n_eps % self.logger_save_freq == 0: 
                cur_time = time.time()
                if self.log_env_info:
                    self.logger.log({
                        "train/ep_rewards": self.ep_rews, 
                        "train/terminated": terminated, 
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                        "train/env_info": self.env_info_fn(self.env_info), 
                        } | metrics, self.n_eps
                    )
                    self.env_info = []
                else: 
                    self.logger.log({
                        "train/ep_rewards": self.ep_rews, 
                        "train/terminated": terminated, 
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                    } | metrics, 
                    self.n_eps
                    )
                
                self.logger_train_update_speed = cur_time

            # reset values
            self.ep_rews = 0
            self._state = None # Force env to reset
            self.reset_metrics()


    
    def update(self, skip_update=None) -> bool:
        '''
        update agents and log time
        returns bool if agents are updated
        '''
        if (skip_update is None or skip_update is True) and self.skip_update(): 
            return False

        _start_update_time = time.time()

        # sample buffer
        # inumerate through data and update
        # self.sample_data()
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

    def eval(self, log=True, return_eval_metrics=False):
        if self.render_evals:
            self.env.change_render_mode('rgb_array')
        state, _ = self.env.reset()
        eval_ep_rews = 0
        length = 0

        eval_renderings = []

        while True:
            action, logits = self.get_action(state, is_training=False)
            if self.render_evals: 
                state, reward, terminated, truncated, _, render = self.env.step(action)
            else:
                state, reward, terminated, truncated, _ = self.env.step(action)
            eval_ep_rews += reward
            eval_metrics = self.get_metrics(logits, logger_dir="eval/")
            length += 1

            if self.render_evals:
                # Append render to eval_renderings shape: (t, height, width, channels)
                eval_renderings.append(render)

            if terminated or truncated:
                break
        

        # Reshape eval_renderings to (t, channels, height, width)
        if log: 
            eval_renderings = np.array(eval_renderings)
            if eval_renderings.ndim == 3:
                # if only single frame, expand dim
                eval_renderings = np.expand_dims(eval_renderings, axis=0)

            elif eval_renderings.ndim == 2:
                # If only single grayscale frame, expand dims
                eval_renderings = np.expand_dims(eval_renderings, axis=-1)
                eval_renderings = np.expand_dims(eval_renderings, axis=0)
            if eval_renderings.ndim == 4:
                eval_render_T = np.transpose(eval_renderings, (0, 3, 1, 2))
                wandb_video = wandb.Video(eval_render_T, 
                                    fps=self.fps, 
                                    format="mp4", 
                                    caption=f"{self.env.__class__.__name__} Render: Eval at episode: {self.n_eps}, \
                                        rew: {eval_ep_rews}")
                self.logger.log({"eval/video": wandb_video}, self.n_eps)
            elif eval_renderings.ndim == 1:
                # if no video collected
                pass
            else: 
                print(f"Error: Final rendering array has unexpected dimensions: {eval_renderings.ndim}")

            self.logger.log({
                    "eval/ep_rewards": eval_ep_rews, 
                    "eval/length": length
                } | eval_metrics, self.n_eps) 
        
        if self.save_best_model:
            self.cur_score = (self.best_model_exp_moving_avg * self.cur_score) + \
                             ((1 - self.best_model_exp_moving_avg) * eval_ep_rews)
            if self.cur_score > self.save_threshold: 
                if os.path.exists(self.model_save_path) is False:
                    os.makedirs(self.model_save_path)
                self.save_agents(path_name=f"{self.model_save_path}/" + self.name)
                self.save_threshold = self.cur_score
        if return_eval_metrics:
            return {
                "final/ep_rewards": eval_ep_rews, 
                "final/length": length
            } 

    def cleanup(self):
        self.logger.close()
        self.buffer.clear()
        self.env.close()


    