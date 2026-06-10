from src.train.base_trainer import BaseTrainer

from src.buffers.base_buffer import BaseBuffer
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.loggers.wandb_logger import WandBLogger
from src.envs.mod_base_gym_env import modBaseGymEnv

from src.agents.base_agent import BaseAgent
from src.agents.off_policy_agents.value_agent import ValueAgent
from src.agents.on_policy_agents.policy_agent import PolicyAgent
from src.agents.on_policy_agents.A2C import A2CAgent

import math
from typing import Dict, List
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import wandb

class MARLTrainer(BaseTrainer): 
    def __init__(self, 
                agents: Dict, 
                env: modBaseGymEnv, # Custom modified base gym
                logger_config: dict, 
                logger: str = "WandBLogger",
                logger_save_freq: int = 1, 
                eval_freq: int = 0,
                model_save_freq: int = 1000,
                model_save_path: str = './src/trained_agents/',
                save_best_model: bool = False,
                best_model_exp_moving_avg: float = 0.95,
                log_env_info: bool = False,
                env_info_fn = None,
                render_evals = True,
                fps: int = 5,
                shared_buffer = True, 
                buffer: BaseBuffer = None,
                equal_batch_size: bool = True, 
                batch_size: int = 256,
                update_same_time: bool = True, 
                model_update_freq: int = 1, 
                n_update_steps: int = 1
                ):
        '''
        Custom MARL Trainer
        Args:
            agents (Dict): Dictionary of Agents with respective configs.
            Dict[List[BaseAgent, BaseBuffer, bool, int, int, int]] | Dict[List[BaseAgent, BaseBuffer, bool, int, int, int, int]]
                On policy agent list config: agent, buffer, is_on_policy, batch_size, n_epochs, team
                Off policy agent list config: agent, buffer, is_on_policy, batch_size, model_update_freq, n_update_steps, team
                if team = 0 then no associated team and will just take 0th reward
        '''
        self.name = logger_config["name"]

        # Initialize environments: 
        self.env = env
        #TODO: implement parameters to have custom observations for each agent

        self._state = None
        self.ep_rews = None
        self.ep_train_certainty = None
        self.log_env_info = log_env_info
        self.env_info_fn = env_info_fn
        self.env_info = None

        if log_env_info and env_info_fn is None:
            raise TypeError("log_env_info defined, but env_info_fn not defined. ")
        if log_env_info:
            print("Warning: logging environment info. " \
            "Assuming info is being passed back as appropriate type to be passed through env_info_fn and logged")
        
        # Initialize eval and update settings
        self.eval_freq = eval_freq
        self.save_best_model = save_best_model
        if self.save_best_model:
            self.cur_score = 0.0
            self.best_model_exp_moving_avg = best_model_exp_moving_avg
            self.save_threshold = -math.inf

        self.n_train_step = 0
        
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        self.n_updates = 0
        self.n_eps = 0

        self.render_evals = render_evals
        self.fps = fps
        if self.render_evals:
            print(f"Saving eval renderings. fps set to {self.fps}")

        
        # Initialize Agents: 
        self.agent_configs = agents
        if shared_buffer: 
            self.buffer = buffer
            assert self.buffer is not None, "buffer must be not None if there's a shared buffer"
        

        if equal_batch_size and self.buffer is not None: 
            self.batch_size = batch_size
            assert batch_size <= self.buffer.max_size, f"Batch size must be less than or equal to buffer size for on-policy agents. Current buffer size: {self.buffer.max_size}, batch size: {self.batch_size}"
            assert self.buffer.max_size % self.batch_size == 0, "Buffer size must be multiple of batch size for on-policy agents"

        
        if update_same_time: 
            self.model_update_freq = model_update_freq
            self.n_update_steps = n_update_steps


        self.has_same_max_buffer = True
        prev_buffer_max_size = None

        self.agents = []
        print("Agents in this run:", agents)
        self.buffers = []
        self.n_agents = 0
        # Encoding scalar if same policy across all agents. 
        # None --> No, 1 --> all on policy, 2 --> all off policy
        self.is_same_policy = None 
        for key, data in agents.items(): 
            # if is on policy
            if data[2]: 
                # on polict agent
                agent, buffer, is_on_policy, batch_size, n_epochs = data
                assert batch_size > 0, "Batch size must be positive for on-policy agents"
                assert self.eval_freq > 0, "Evaluation frequency must be positive for on-policy agents"
                if not equal_batch_size: 
                    assert batch_size <= self.buffer.max_size, f"Batch size must be less than or equal to buffer size for on-policy agents. Current buffer size: {self.buffer.max_size}, batch size: {self.batch_size}"
                    assert self.buffer.max_size % self.batch_size == 0, "Buffer size must be multiple of batch size for on-policy agents"
                if isinstance(agent, A2CAgent):
                    assert n_epochs == 1, "Number of epochs must be 1 for A2C agents"
                else:
                    assert n_epochs > 0, "Number of epochs must be positive for on-policy agents"
                if self.is_same_policy is None: 
                    self.is_same_policy = 1
                elif self.is_same_policy == 2:
                    self.is_same_policy = False
            else: 
                # off policy agent
                agent, buffer, is_on_policy, batch_size, model_update_freq, n_update_steps = data
                assert batch_size > 0, "Batch size must be positive for off-policy agents"
                assert self.n_update_steps > 0, "Number of update steps must be positive for off-policy agents"
                assert self.model_update_freq > 0, "Model update frequency must be positive for off-policy agents"
                if self.is_same_policy is None: 
                    self.is_same_policy = 2
                elif self.is_same_policy == 1:
                    self.is_same_policy = False
            
            self.agents.append(agent)
            self.buffers.append(buffer)
            self.n_agents += 1
            if buffer.max_size != prev_buffer_max_size and prev_buffer_max_size is not None: 
                self.has_same_max_buffer = False
            prev_buffer_max_size = buffer.max_size
                
        self.is_off_policy = self.is_same_policy == 2
        self.is_on_policy = self.is_same_policy == 1

        # Initialize logger: 
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
        self.init_logger()
        self.logger_save_freq = logger_save_freq
        self.logger_train_update_speed = 0

    def init_logger(self):
        for agent in self.agents: 
            models = agent.get_models()
            loss_fns = agent.get_loss_fns()
            zipped_data = zip(models, loss_fns)

            # Track all models: gradients and parameters 
            for i, (model, loss_fn) in enumerate(zipped_data):
                self.logger.watch_model(model, criterion=loss_fn, idx=i)
    
    def train(self, n_steps: int, run_eval: int = 0) -> None: 
        if self.is_same_policy is None: 
            self.mixed_agent_train(n_steps, run_eval)
        else: 
            for t in tqdm(range(n_steps)): 
                self.run_step()

                if self.is_off_policy and t % self.model_update_freq == 0 and t > 0: 
                    self.update()

                if t % self.eval_freq == 0 and t > 0: 
                    self.eval()
                
                if t % self.model_save_freq == 0 and t > 0:
                    if os.path.exists(self.model_save_path) is False:
                        os.makedirs(self.model_save_path)
                    for agent in self.agents: 
                        agent.save_model(f"{self.model_save_path}/checkpoint_{self.n_updates}" + self.name)

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
    
    def mixed_agent_train(self, n_steps: int, run_eval: int = 0) -> None: 
        #TODO: implement mixed policy agent training
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

    def run_step(self):
        if self.env.render_mode is not None: 
            self.env.change_render_mode(None)
        
       
        if self.log_env_info:
            self.env_info = []

        if self._state is None: 
            self._state, info = self.env.reset()

        # Select actions
        actions = [0 for _ in range(self.n_agents)]
        if self.is_on_policy:
            logits = [0 for _ in range(self.n_agents)]
            for i, agent in enumerate(self.agents): 
                action, agent_logits = agent.select_action(self._state.to(agent.device), is_training=True)
                actions[i] = action
                logits[i] = agent_logits
        elif self.is_off_policy:
            for i, agent in enumerate(self.agents): 
                action = agent.select_action(self._state.to(agent.device), is_training=True)
                actions[i] = action
        
        #TODO: handle multiple rewards...
        next_state, reward, terminated, truncated, info = self.env.step(actions)
        # TODO: What if "done"/terminated or truncated is per agent? 
        done = terminated or truncated

        if self.ep_rews is None: 
            self.ep_rews = [0 for _ in range(len(reward))]
        if self.ep_train_certainty is None:
            self.ep_train_certainty = [0 for _ in range(self.n_agents)]

        if self.is_on_policy:
            # TODO: Add option for a single value estimate function
            # If on-policy add value estimates to buffer
            state_values = [0 for _ in range(self.n_agents)]
            if isinstance(self._state, torch.Tensor):
                for i, agent in enumerate(self.agents): 
                    state_value = agent.find_value(self._state.to(agent.device))
                    state_values[i] = state_value
            elif isinstance(self._state, np.ndarray):
                for i, agent in enumerate(self.agents): 
                    state_value = agent.find_value(torch.tensor(self._state, dtype=torch.float32).to(agent.device))
                    state_values[i] = state_value
            else:
                raise TypeError(f"Unknown state type returned from env: {type(self._state)}")
            
            for i, (key, configs) in enumerate(self.agent_configs.items()): 
                # add to buffers if buffer isn't already finalized
                if not configs[1].is_buffer_full: 
                    configs[1].add({
                        "state": self._state,
                        "action": actions[i],
                        "reward": reward[configs[-1]],
                        "log_prob": logits[i],
                        "done": done,
                        "value": state_values[i].detach().cpu().item(),
                    })

            call_update = True
            # Only call update if all buffers are full
            # Can't shortcut by checking a singular buffer even if same size (agents might have different play times)
            for i, configs in enumerate(self.agent_configs.values()): 
                agent, buffer, is_on_policy, batch_size, n_epochs = configs
                if not buffer.is_buffer_full: 
                    call_update = False
                else: 
                    # buffer full, so finalize buffer
                    self.buffer.finalize_buffer(agent.find_value(next_state.to(agent.device)).detach().cpu().item())
                
            if call_update:
                self.update()
            
            for buffer in self.buffers:
                buffer.clear()

        else: 
            for i, (key, configs) in enumerate(self.agent_configs.items()): 
                # else if off-policy add normal state values
                configs[1].add({
                    "state": self._state, 
                    "action": actions[i],
                    "reward": reward[configs[-1]],
                    "done": done, 
                })

        self._state = next_state

        for i in range(len(reward)): 
            self.ep_rews[i] += reward[i]
        if self.log_env_info:
            self.env_info.append(info)

        # Metric to track certainty in guesses
        for i in range(self.n_agents):
            self.ep_train_certainty[i] += self.calc_certainty(logits[i])

        self.n_train_step += 1
        # Log episodic values if done
        if done:
            self.n_eps += 1
            if self.n_eps % self.logger_save_freq == 0: 
                cur_time = time.time()

                agent_logs = {}
                for i, (key, data) in enumerate(self.agent_configs.items()):
                    agent_logs[f"train/{key}/ep_rewards"] = self.ep_rews[i]
                    agent_logs[f"train/{key}/ep_certainty"] = self.ep_train_certainty[i]


                if self.log_env_info:
                    self.logger.log({
                        "train/sum_rewards": sum(self.ep_rews), 
                        "train/terminated": terminated, 
                        "train/sum_ep_certainty": sum(self.ep_train_certainty),
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                        "train/env_info": self.env_info_fn(self.env_info), 
                        **agent_logs
                        }, self.n_eps
                    )
                    self.env_info = []
                else: 
                    self.logger.log({
                        "train/sum_rewards": sum(self.ep_rews), 
                        "train/terminated": terminated, 
                        "train/sum_ep_certainty": sum(self.ep_train_certainty),
                        "train/logging_per_sec": 1 / (cur_time - self.logger_train_update_speed),
                        **agent_logs
                    }, 
                    self.n_eps
                    )
                
                self.logger_train_update_speed = cur_time

            # reset values
            self.ep_rews = None
            self.ep_train_certainty = None
            self._state = None # Force env to reset

    def calc_certainty(self, logits: torch.Tensor) -> float:
        '''
        Find "certainty": calculate "certainty" by using entropy H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)
        '''
        #TODO logits is not necessarily the same across agents, need to handle different action spaces and output formats
        return -math.exp(logits.item()) * logits.item()


    def update(self):
        # if self._start_update_time is None: 
        _start_update_time = time.time()
        cur_updates = 0

        if self.is_on_policy:
            for agent_config in self.agent_configs.values(): 
                agent, buffer, is_on_policy, batch_size, n_epochs = agent_config
                #TODO add epoch looping
                # On-policy update
                sample_data = buffer.sample(batch_size, clear_buffer=True)
                if isinstance(sample_data, list) and isinstance(buffer, TorchTensorBuffer):
                    # assuming if sample_data is single list then each item will hold dict with values lists of experiences
                    for i, experience in enumerate(sample_data):
                        update_metrics = agent.update(
                            states=experience['state'], 
                            returns=experience['return'], 
                            advantages=experience['advantage'], 
                            actions=experience['action'],
                            identifier=i
                        )
                else:
                    raise TypeError("Unexpected sample data type during off-policy update or unexpected buffer type")
            self.n_updates += len(sample_data)
            cur_updates = len(sample_data)

            # self.logger.log(update_metrics, self.n_eps)
        elif self.is_off_policy:
            # Off-policy update
            for agent_config in self.agent_configs: 
                agent, buffer, is_on_policy, batch_size, model_update_freq, n_update_steps = agent_config
                sample_data = buffer.sample(batch_size, clear_buffer=False)
                for i in range(n_update_steps):
                    update_metrics = agent.update(
                        state=sample_data['state'], 
                        action=sample_data['action'], 
                        reward=sample_data['reward'], 
                        done=sample_data['done'], 
                        identifier=i
                    )
            self.n_updates += n_update_steps
            cur_updates = n_update_steps
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
        eval_ep_rews = None
        eval_ep_certainty = None
        length = 0

        eval_renderings = []

        while True:
            # Select actions and get logits
            actions = [0 for _ in range(self.n_agents)]
            if self.is_on_policy:
                logits = [0 for _ in range(self.n_agents)]
                for i, agent in enumerate(self.agents): 
                    action, agent_logits = agent.select_action(state.to(agent.device), is_training=False)
                    actions[i] = action
                    logits[i] = agent_logits
            elif self.is_off_policy:
                for i, agent in enumerate(self.agents): 
                    action = agent.select_action(state.to(agent.device), is_training=False)
                    actions[i] = action

            if self.render_evals: 
                state, reward, terminated, truncated, _, render = self.env.step(actions)
            else:
                state, reward, terminated, truncated, _ = self.env.step(actions)
            
            if eval_ep_rews is None: 
                eval_ep_rews = [0 for _ in range(len(reward))]
            if eval_ep_certainty is None and self.is_on_policy:
                eval_ep_certainty = [0 for _ in range(self.n_agents)]
            
            # Update rewards and certainty (if necessary)
            for i in range(len(reward)): 
                eval_ep_rews[i] += reward[i]
            if self.is_on_policy: 
                for i in range(self.n_agents):
                    eval_ep_certainty[i] += self.calc_certainty(logits[i])
            
            
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

            agent_logs = {}
            for i, (key, data) in enumerate(self.agent_configs.items()):
                agent_logs[f"eval/{key}/ep_rewards"] = eval_ep_rews[i]
                agent_logs[f"eval/{key}/ep_certainty"] = eval_ep_certainty[i]


            if self.log_env_info:
                self.logger.log({
                    "eval/sum_rewards": sum(eval_ep_rews), 
                    "eval/terminated": terminated, 
                    "train/sum_ep_certainty": sum(eval_ep_certainty),
                    **agent_logs
                    }, self.n_eps
                )
        
        if self.save_best_model:
            # TODO: Find way to implement saving best model
            # self.cur_score = (self.best_model_exp_moving_avg * self.cur_score) + \
            #                  ((1 - self.best_model_exp_moving_avg) * eval_ep_rews)
            # if self.cur_score > self.save_threshold: 
            #     if os.path.exists(self.model_save_path) is False:
            #         os.makedirs(self.model_save_path)
            #     self.agent.save_model(f"{self.model_save_path}/" + self.name)
            #     self.save_threshold = self.cur_score
            pass
        
        if return_eval_metrics:
            return {
                "final/ep_rewards": sum(eval_ep_rews), 
                "final/ep_certainty": sum(eval_ep_certainty),
                "final/length": length
            }

    def cleanup(self):
        self.logger.close()
        self.buffer.clear()
        self.env.close()


