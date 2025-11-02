from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    env_name: str = "GridEnv"
    env_size: int = 5
    max_steps_per_episode: int = 50
    render_mode: str = "human"  # Options: "human", "rgb_array", None

@dataclass
class LoggerConfig:
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
    model: str = "A2C"
    policy_network: str = "NeuralNetwork"
    critic_network: str = "NeuralNetwork"
    hidden_network_size: list[int] = field(default_factory=lambda: [64])
    learning_rate: float = 0.0001
    gamma: float = 0.99
    grad_clip: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    optimizer: str = "ADAM"

@dataclass
class BufferConfig:
    type: str = "OnPolicy"
    name: str = "TorchTensorBuffer" 
    buffer_size: int = 100_000
    batch_size: int = 64
    gae_lambda: float = 0.95
    gamma: float = 0.99

@dataclass
class TrainingParams:
    batch_size: int = 500
    eval_freq: int = 100 
    model_update_freq: int = 1000
    n_update_steps: int = 1
    model_save_freq: int = 1000
    model_save_path: str = './src/trained_agent'
    save_best_model: bool = True
    log_env_info: bool = False
    render_evals: bool = True

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
    logger: LoggerConfig = LoggerConfig(wandb_config={})
    agent: AgentConfig = AgentConfig()
    training: TrainingParams = TrainingParams()
    buffer: BufferConfig = BufferConfig()