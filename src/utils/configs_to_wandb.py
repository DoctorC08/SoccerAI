from src.utils.config import EnvConfig, AgentConfig, BufferConfig, TrainingParams, TrainerConfig

def ConfigsToWandb(env_config: EnvConfig, agent_config: AgentConfig, buffer_config: BufferConfig, training_params: TrainingParams, trainer_params: TrainerConfig = None):
    wandb_config = {}
    wandb_config.update(env_config.to_wandb_config())
    wandb_config.update(agent_config.to_wandb_config())
    wandb_config.update(buffer_config.to_wandb_config())
    wandb_config.update(training_params.to_wandb_config())
    if trainer_params is not None:
        wandb_config.update(trainer_params.to_wandb_config())
    return wandb_config