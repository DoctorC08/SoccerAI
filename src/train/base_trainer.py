import math
from abc import ABC, abstractmethod


from src.buffers.base_buffer import BaseBuffer

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
    
    @abstractmethod
    def init_logger(self) -> None:
        pass
    
    @abstractmethod
    def train(self, n_steps: int, run_eval: int = 0) -> None: 
        pass

    @abstractmethod
    def run_step(self):
        pass
    
    @abstractmethod
    def update(self):
       pass

    @abstractmethod
    def eval(self, log=True, return_eval_metrics=False):
        pass

    def cleanup(self):
        self.logger.close()
        self.buffer.clear()
        self.env.close()


    