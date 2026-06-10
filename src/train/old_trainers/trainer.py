import math
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import wandb

from src.buffers.base_buffer import BaseBuffer
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.loggers.wandb_logger import WandBLogger
from src.envs.mod_base_gym_env import modBaseGymEnv

from src.agents.base_agent import BaseAgent
from src.agents.off_policy_agents.value_agent import ValueAgent
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent

from src.train.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, 
                agent: BaseAgent, 
                buffer: BaseBuffer,
                env: modBaseGymEnv, # Custom modified base gym
                logger_config: dict, 
                logger: str = "WandBLogger",
                logger_save_freq: int = 1, 
                batch_size: int = 0,
                eval_freq: int = 0,
                model_update_freq: int = 1, # Only use for off-policy agents
                n_update_steps: int = 1, # This only use for off-policy agents
                n_epochs: int = 1, # This is only used for some on-policy agents
                model_save_freq: int = 1000,
                model_save_path: str = './src/trained_agents/',
                save_best_model: bool = True,
                best_model_exp_moving_avg: float = 0.95,
                log_env_info: bool = False,
                env_info_fn = None,
                render_evals = True,
                fps: int = 5,
            ):
        
        self.name = logger_config["name"]
        self.agent = agent
        self.buffer = buffer
        self.env = env
        if logger == "WandBLogger":
            self.logger = WandBLogger(
                project = logger_config["project"], 
                name = logger_config["name"], 
                config = logger_config["config"], 
                reinit = logger_config["reinit"],
                sweep = logger_config["sweep"],
            )
        else: 
            raise LookupError(f"Unknown logger inputed: {logger}")
        
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.save_best_model = save_best_model
        if self.save_best_model:
            self.cur_score = 0.0
            self.best_model_exp_moving_avg = best_model_exp_moving_avg
            self.save_threshold = -math.inf

        self.n_train_step = 0
        self.model_update_freq = model_update_freq
        self.n_update_steps = n_update_steps
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        self.n_updates = 0
        self.n_eps = 0

        # Determine if on-policy or off-policy agent
        self.is_off_policy = isinstance(self.agent, ValueAgent) 
        self.is_on_policy = isinstance(self.agent, PolicyAgent)

        if self.is_on_policy:
            assert self.batch_size > 0, "Batch size must be positive for on-policy agents"
            assert self.eval_freq > 0, "Evaluation frequency must be positive for on-policy agents"
            assert self.batch_size <= self.buffer.max_size, f"Batch size must be less than or equal to buffer size for on-policy agents. Current buffer size: {self.buffer.max_size}, batch size: {self.batch_size}"
            assert self.buffer.max_size % self.batch_size == 0, "Buffer size must be multiple of batch size for on-policy agents"
            if isinstance(self.agent, A2CAgent):
                assert n_epochs == 1, "Number of epochs must be 1 for A2C agents"
            else:
                assert n_epochs > 0, "Number of epochs must be positive for on-policy agents"

        elif self.is_off_policy:
            assert self.n_update_steps > 0, "Number of update steps must be positive for off-policy agents"
            assert self.model_update_freq > 0, "Model update frequency must be positive for off-policy agents"
        else: 
            raise TypeError("Agent must be either on-policy or off-policy type. Unknown type used")


        self.init_logger()
        self.logger_save_freq = logger_save_freq
        self.logger_train_update_speed = 0

        self._state = None
        self.ep_rews = 0.0 
        self.ep_train_certainty = 0.0
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


    def init_logger(self):
        models = self.agent.get_models()
        loss_fns = self.agent.get_loss_fns()
        zipped_data = zip(models, loss_fns)

        # Track all models: gradients and parameters 
        for i, (model, loss_fn) in enumerate(zipped_data):
            self.logger.watch_model(model, criterion=loss_fn, idx=i)
    
    def train(self, n_steps: int, run_eval: int = 0) -> None: 
        for t in tqdm(range(n_steps)): 
            self.run_step()

            if self.is_off_policy and t % self.model_update_freq == 0 and t > 0: 
                self.update()

            if t % self.eval_freq == 0 and t > 0: 
                self.eval()
            
            if t % self.model_save_freq == 0 and t > 0:
                if os.path.exists(self.model_save_path) is False:
                    os.makedirs(self.model_save_path)
                self.agent.save_model(f"{self.model_save_path}/checkpoint_{self.n_updates}" + self.name)

        if run_eval: 
            eval_metrics = {}
            for _ in range(run_eval):
                cur_metrics = self.eval(log=False, return_eval_metrics=True)
                for key, value in cur_metrics.items():
                    eval_metrics[key] = eval_metrics.get(key, 0) + value

            # average values
            for key in eval_metrics.keys():
                eval_metrics[key] /= run_eval
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
        if self.env.render_mode is not None: 
            self.env.change_render_mode(None)
        if self.ep_rews is None: 
            self.ep_rews = 0
        if self.ep_train_certainty is None:
            self.ep_train_certainty = 0
        if self.log_env_info:
            self.env_info = []

        if self._state is None: 
            self._state, info = self.env.reset()

        if self.is_on_policy:
            action, logits = self.agent.select_action(self._state.to(self.agent.device), is_training=True)
        elif self.is_off_policy:
            action = self.agent.select_action(self._state, is_training=True)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self.is_on_policy:
            # If on-policy add value estimates to buffer
            if isinstance(self._state, torch.Tensor):
                state_value = self.agent.find_value(self._state.to(self.agent.device))
            elif isinstance(self._state, np.ndarray):
                state_value = self.agent.find_value(torch.tensor(self._state, dtype=torch.float32).to(self.agent.device))
            else:
                raise TypeError(f"Unknown state type returned from env: {type(self._state)}")
            self.buffer.add({
                "state": self._state,
                "action": action,
                "reward": reward,
                "done": done,
                "log_prob": logits,
                "value": state_value.detach().cpu().item(),
            })
            if self.buffer.is_buffer_full:
                # print("Buffer full, finalizing buffer")
                self.buffer.finalize_buffer(self.agent.find_value(next_state.to(self.agent.device)).detach().cpu().item())
                self.update()
                self.buffer.clear()
        else: 
            # else if off-policy add normal state values
            self.buffer.add({
                "state": self._state, 
                "action": action,
                "reward": reward,
                "done": done, 
            })
        self._state = next_state

        self.ep_rews += reward
        if self.log_env_info:
            self.env_info.append(info)

        # Metric to track certainty in guesses
        self.ep_train_certainty += self.calc_certainty(logits)

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
                        "train/ep_certainty": self.ep_train_certainty,
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                        "train/env_info": self.env_info_fn(self.env_info), 
                        }, self.n_eps
                    )
                    self.env_info = []
                else: 
                    self.logger.log({
                        "train/ep_rewards": self.ep_rews, 
                        "train/terminated": terminated, 
                        "train/ep_certainty": self.ep_train_certainty,
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                    }, 
                    self.n_eps
                    )
                
                self.logger_train_update_speed = cur_time

            # reset values
            self.ep_rews = 0
            self.ep_train_certainty = 0
            self._state = None # Force env to reset

    def calc_certainty(self, logits: torch.Tensor) -> float:
        '''
        Find "certainty": calculate "certainty" by using entropy H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)
        '''
        return -math.exp(logits.item()) * logits.item()


    def update(self):
        # if self._start_update_time is None: 
        _start_update_time = time.time()
        cur_updates = 0

        if self.is_on_policy:
            # On-policy update
            sample_data = self.buffer.sample(self.batch_size, clear_buffer=True)
            if isinstance(sample_data, list) and isinstance(self.buffer, TorchTensorBuffer):
                # assuming if sample_data is single list then each item will hold dict with values lists of experiences
                for i, experience in enumerate(sample_data):
                    update_metrics = self.agent.update(states=experience['state'], 
                                                       returns=experience['return'], 
                                                       advantages=experience['advantage'], 
                                                       actions=experience['action'],
                                                       identifier=i)
                    self.n_updates += 1
                    cur_updates += 1

                    # self.logger.log(update_metrics, self.n_eps)
            else:
                raise TypeError("Unexpected sample data type during off-policy update or unexpected buffer type")
        elif self.is_off_policy:
            # Off-policy update
            sample_data = self.buffer.sample(self.batch_size, clear_buffer=False)
            for i in range(self.n_update_steps):
                update_metrics = self.agent.update(
                    state=sample_data['state'], 
                    action=sample_data['action'], 
                    reward=sample_data['reward'], 
                    done=sample_data['done'], 
                    identifier=i
                )
                self.n_updates += 1
                cur_updates += 1
                # self.logger.log(update_metrics, self.n_eps)


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

    def eval(self, log=True, return_eval_metrics=False):
        if self.render_evals:
            self.env.change_render_mode('rgb_array')
        state, _ = self.env.reset()
        eval_ep_rews = 0
        eval_ep_certainty = 0
        length = 0

        eval_renderings = []

        while True:
            action, logits = self.agent.select_action(state, is_training=False)
            if self.render_evals: 
                state, reward, terminated, truncated, _, render = self.env.step(action)
            else:
                state, reward, terminated, truncated, _ = self.env.step(action)
            eval_ep_rews += reward
            eval_ep_certainty += self.calc_certainty(logits)
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
                    "eval/ep_certainty": eval_ep_certainty,
                    "eval/length": length
                }, self.n_eps) 
        
        if self.save_best_model:
            self.cur_score = (self.best_model_exp_moving_avg * self.cur_score) + \
                             ((1 - self.best_model_exp_moving_avg) * eval_ep_rews)
            if self.cur_score > self.save_threshold: 
                if os.path.exists(self.model_save_path) is False:
                    os.makedirs(self.model_save_path)
                self.agent.save_model(f"{self.model_save_path}/" + self.name)
                self.save_threshold = self.cur_score
        if return_eval_metrics:
            return {
                "final/ep_rewards": eval_ep_rews, 
                "final/ep_certainty": eval_ep_certainty,
                "final/length": length
            }

    def cleanup(self):
        self.logger.close()
        self.buffer.clear()
        self.env.close()