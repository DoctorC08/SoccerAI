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

@dataclass
class LoggerConfig:
    logger: str = "WandBLogger"
    project: str = "SoccerAI"
    name: str = "DefaultRun"
    reinit: bool = False
    wandb_config: dict = None  # Additional WandB config parameters

    def __post_init__(self):
        self.logger_config = {
            "project": self.project,
            "name": self.name,
            "reinit": self.reinit,
            "config": self.wandb_config if self.wandb_config is not None else {},
        }
        if self.wandb_config is None:
            print("Warning: config is not provided, using empty config.")
            print("Current config:", self.wandb_config)

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

@dataclass
class TrainingParams:
    total_training_steps: int = 100_000
    batch_size: int = 500
    eval_freq: int = 100 
    model_update_freq: int = 1000
    n_update_steps: int = 1
    n_epochs: int = 1
    model_save_freq: int = 1000
    model_save_path: str = './src/trained_agent'
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

@dataclass 
class Config:
    device: str = "cpu"
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    training: TrainingParams = TrainingParams()
    buffer: BufferConfig = BufferConfig()
    logger: LoggerConfig = LoggerConfig(wandb_config={})