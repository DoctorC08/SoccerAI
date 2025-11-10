from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    env_name: str = "GridEnv"
    env_size: int = 5
    max_steps_per_episode: int = 50
    render_mode: str = "human"  # Options: "human", "rgb_array", None

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
class LoggerConfig:
    logger: str = "WandBLogger"
    project: str = "SoccerAI"
    name: str = "DefaultRun"
    reinit: bool = False
    wandb_config: dict = None  # Additional WandB config parameters
    sweep: bool = False
    logger_save_freq: int = 1

    def __post_init__(self):
        self.logger_config = {
            "project": self.project,
            "name": self.name,
            "reinit": self.reinit,
            "config": self.wandb_config if self.wandb_config is not None else {},
            "sweep": self.sweep,
        }
        if self.wandb_config is None:
            print("Warning: config is not provided, using empty config.")
            print("Current config:", self.wandb_config)

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


# =====================
# BUFFER CONFIG
# =====================
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
class TrainingParams:
    total_training_steps: int = 100_000
    batch_size: int = 500
    eval_freq: int = 100
    model_update_freq: int = 1000
    n_update_steps: int = 1
    n_epochs: int = 1
    model_save_freq: int = 1000
    model_save_path: str = "./src/trained_agent"
    save_best_model: bool = True
    best_model_exp_moving_avg: float = 0.99
    log_env_info: bool = False
    env_info_fn: callable = None
    render_evals: bool = True
    fps: int = 5

    def to_wandb_config(self):
        return {
            "batch_size": self.batch_size,
            "eval_freq": self.eval_freq,
            "model_update_freq": self.model_update_freq,
            "n_update_steps": self.n_update_steps,
            "model_save_freq": self.model_save_freq,
        }

    def __str__(self):
        return (
            f"TrainingParams:\n"
            f"  total_training_steps: {self.total_training_steps}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  eval_freq: {self.eval_freq}\n"
            f"  model_update_freq: {self.model_update_freq}\n"
            f"  n_update_steps: {self.n_update_steps}\n"
            f"  n_epochs: {self.n_epochs}\n"
            f"  model_save_freq: {self.model_save_freq}\n"
            f"  model_save_path: {self.model_save_path}\n"
            f"  save_best_model: {self.save_best_model}\n"
            f"  best_model_exp_moving_avg: {self.best_model_exp_moving_avg}\n"
            f"  log_env_info: {self.log_env_info}\n"
            f"  render_evals: {self.render_evals}\n"
            f"  fps: {self.fps}"
        )


@dataclass
class Config:
    device: str = "cpu"
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    training: TrainingParams = TrainingParams()
    buffer: BufferConfig = BufferConfig()
    logger: LoggerConfig = LoggerConfig(wandb_config={})

    def __str__(self):
        return (
            f"Config:\n"
            f"  device: {self.device}\n"
            f"{self.env}\n"
            f"{self.agent}\n"
            f"{self.training}\n"
            f"{self.buffer}\n"
            f"{self.logger}"
        )
