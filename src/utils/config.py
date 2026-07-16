from dataclasses import dataclass, field
from typing import Any
import torch

from src.eval.base_eval import BaseEval
from src.eval.single_agent_eval import SingleAgentEval

@dataclass
class EnvConfig:
    env_name: str = "GridEnv"
    render_mode: str = "human"  # Options: "human", "rgb_array", None

    def to_wandb_config(self):
        return {
            "env_name": self.env_name,
            "render_mode": self.render_mode,
        }

    def __str__(self):
        return (
            f"EnvConfig:\n"
            f"  env_name: {self.env_name}\n"
            f"  render_mode: {self.render_mode}"
        )

@dataclass
class GridEnvConfig(EnvConfig):
    env_name: str = "GridEnv"
    env_size: int = 5
    max_steps_per_episode: int = 50

    def to_wandb_config(self):
        return {
            "env_name": self.env_name,
            "env_size": self.env_size,
            "max_steps_per_episode": self.max_steps_per_episode,
            "render_mode": self.render_mode,
        }

    def __str__(self):
        return (
            f"EnvConfig:\n"
            f"  env_name: {self.env_name}\n"
            f"  env_size: {self.env_size}\n"
            f"  max_steps_per_episode: {self.max_steps_per_episode}\n"
            f"  render_mode: {self.render_mode}"
        )

@dataclass
class SoccerEnvConfig(EnvConfig):
    env_name: str = "SoccerEnv"
    team_a_size: int = 2
    team_b_size: int = 2
    width: float = 100.0
    height: float = 60.0
    time_step: float = 0.1
    goal_size: float = 20.0
    kf: float = 20.0
    fric: float = 0.85
    bmw: float = 0.5
    pmw: float = 0.2
    max_steps: int = 1000
    random_ball_placement: bool = False
    sim_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_wandb_config(self):
        return {
            "env_name": self.env_name,
            "team_a_size": self.team_a_size,
            "team_b_size": self.team_b_size,
            "width": self.width,
            "height": self.height,
            "time_step": self.time_step,
            "goal_size": self.goal_size,
            "kf": self.kf,
            "fric": self.fric,
            "bmw": self.bmw,
            "pmw": self.pmw,
            "random_ball_placement": self.random_ball_placement,
            "render_mode": self.render_mode,
            "sim_kwargs": self.sim_kwargs,
        }

    def __str__(self):
        return (
            f"SoccerEnvConfig:\n"
            f"  env_name: {self.env_name}\n"
            f"  team_a_size: {self.team_a_size}\n"
            f"  team_b_size: {self.team_b_size}\n"
            f"  width: {self.width}\n"
            f"  height: {self.height}\n"
            f"  time_step: {self.time_step}\n"
            f"  goal_size: {self.goal_size}\n"
            f"  kf: {self.kf}\n"
            f"  fric: {self.fric}\n"
            f"  bmw: {self.bmw}\n"
            f"  pmw: {self.pmw}\n"
            f"  random_ball_placement: {self.random_ball_placement}\n"
            f"  render_mode: {self.render_mode}\n"
            f"  sim_kwargs: {self.sim_kwargs}"
        )

@dataclass
class LoggerConfig:
    logger: str = "WandBLogger"
    project: str = "SoccerAI"
    name: str = "DefaultRun"
    reinit: bool = False
    wandb_config: dict = field(default_factory=dict)  # Additional WandB config parameters
    sweep: bool = False
    logger_save_freq: int = 1


    def __str__(self):
        return (
            f"LoggerConfig:\n"
            f"  logger: {self.logger}\n"
            f"  project: {self.project}\n"
            f"  name: {self.name}\n"
            f"  reinit: {self.reinit}\n"
            f"  sweep: {self.sweep}\n"
            f"  wandb_config: {self.wandb_config}\n"
            f"  logger_save_freq: {self.logger_save_freq}"
        )


@dataclass
class AgentConfig:
    on_policy: bool = True
    model: str = "A2C"
    policy_network: str = "NeuralNetwork"
    critic_network: str = "NeuralNetwork"

    transfer_learning: bool = False
    model_load_path: str = ""

    hidden_network_size: list[int] = field(default_factory=lambda: [64])

    learning_rate: float = 0.0001
    gamma: float = 0.99
    grad_clip: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    optimizer: str = "ADAM"
    
    batch_size: int = 64

    model_update_freq: int = 1000
    n_update_steps: int = 1

    def to_wandb_config(self):
        return {
            "model": self.model,
            "policy_network": self.policy_network,
            "critic_network": self.critic_network,
            "transfer_learning": self.transfer_learning,
            "model_load_path": self.model_load_path,
            "hidden_network_size": self.hidden_network_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "grad_clip": self.grad_clip,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "optimizer": self.optimizer,
        }

    def __str__(self):
        return (
            f"AgentConfig:\n"
            f"  on_policy: {self.on_policy}\n"
            f"  model: {self.model}\n"
            f"  policy_network: {self.policy_network}\n"
            f"  critic_network: {self.critic_network}\n"
            f"  transfer_learning: {self.transfer_learning}\n"
            f"  model_load_path: {self.model_load_path}\n"
            f"  hidden_network_size: {self.hidden_network_size}\n"
            f"  learning_rate: {self.learning_rate}\n"
            f"  gamma: {self.gamma}\n"
            f"  grad_clip: {self.grad_clip}\n"
            f"  value_loss_coef: {self.value_loss_coef}\n"
            f"  entropy_coef: {self.entropy_coef}\n"
            f"  optimizer: {self.optimizer}"
        )


@dataclass
class BufferConfig:
    type: str = "OnPolicy"
    name: str = "TorchTensorBuffer"
    buffer_size: int = 100_000
    batch_size: int = 64
    gae_lambda: float = 0.95
    gamma: float = 0.99

    def to_wandb_config(self):
        return {
            "type": self.type,
            "name": self.name,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
        }

    def __str__(self):
        return (
            f"BufferConfig:\n"
            f"  type: {self.type}\n"
            f"  name: {self.name}\n"
            f"  buffer_size: {self.buffer_size}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  gae_lambda: {self.gae_lambda}\n"
            f"  gamma: {self.gamma}"
        )

@dataclass
class EvalParams: 
    evaluator: BaseEval = SingleAgentEval
    fps: int = 5
    eval_freq: int = 100

    def to_wandb_config(self):
        return {
            "evaluator": self.evaluator,
        }

    def __str__(self):
        return (
            f"EvalParams:\n"
            f"  evaluator: {self.evaluator}\n"
            )

@dataclass
class TrainingParams:
    total_training_steps: int = 100_000
    model_save_freq: int = 1000
    model_save_path: str = "./src/trained_agent"
    save_best_model: bool = True
    best_model_exp_moving_avg: float = 0.99
    log_env_info: bool = False
    env_info_fn: callable = None #TODO find data type for callable functions
    render_evals: bool = True
    n_envs: int = 5
    n_epochs: int = 1

    def to_wandb_config(self):
        return {
            "total_training_steps": self.total_training_steps, 
            "model_save_freq": self.model_save_freq,
            "n_envs": self.n_envs, 
        }

    def __str__(self):
        return (
            f"TrainingParams:\n"
            f"  total_training_steps: {self.total_training_steps}\n"
            f"  model_save_freq: {self.model_save_freq}\n"
            f"  model_save_path: {self.model_save_path}\n"
            f"  save_best_model: {self.save_best_model}\n"
            f"  best_model_exp_moving_avg: {self.best_model_exp_moving_avg}\n"
            f"  log_env_info: {self.log_env_info}\n"
            f"  render_evals: {self.render_evals}\n"
            f"  n_envs: {self.n_envs}"
        )


@dataclass
class Config:
    device: torch.device = torch.device("cpu")
    env: EnvConfig | GridEnvConfig | SoccerEnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    marl_agent_configs: dict = field(default_factory=dict)
    evaluator: EvalParams = field(default_factory=EvalParams)

    def __str__(self):
        return (
            f"Config:\n"
            f"  device: {self.device}\n"
            f"  env: {self.env}\n"
            f"  agent: {self.agent}\n"
            f"  training: {self.training}\n"
            f"  buffer: {self.buffer}\n"
            f"  logger: {self.logger}\n"
        )
