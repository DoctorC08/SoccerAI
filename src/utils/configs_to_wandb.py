from src.config import EnvConfig, AgentConfig, BufferConfig, TrainingParams

def ConfigsToWandb(env_config: EnvConfig, agent_config: AgentConfig, buffer_config: BufferConfig, training_params: TrainingParams):
    wandb_config = {}
    wandb_config.update(env_config.to_wandb_config())
    wandb_config.update(agent_config.to_wandb_config())
    wandb_config.update(buffer_config.to_wandb_config())
    wandb_config.update(training_params.to_wandb_config())
    return wandb_config