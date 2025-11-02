import torch
import wandb
import os

from src.agents.neural_network import NeuralNetwork
from src.agents.on_policy_agents.A2C import A2CAgent
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer
from src.envs.grid_env import GridEnv
from src.loggers.wandb_logger import WandBLogger
from src.train.trainer import Trainer
from config import Config, EnvConfig, LoggerConfig, AgentConfig, TrainingParams, BufferConfig


def run(cfg: Config):
    print("Creating env: GridEnv")
    env = GridEnv(
        size = cfg.env.env_size, 
        terminating_step= cfg.env.max_steps_per_episode,
        render_mode = cfg.env.render_mode,
    )
    print("Creating buffer:", cfg.buffer.name)
    buffer = TorchTensorBuffer(
        max_size=cfg.buffer.buffer_size,
        device=cfg.device,
        batch_size=cfg.buffer.batch_size,
        gamma=cfg.buffer.gamma,
        gae_lambda=cfg.buffer.gae_lambda
    )

    print("Creating agent:", cfg.agent.model)
    state_size=env.observation_space.shape[0], 
    action_size=env.action_space.n,

    policy_network = NeuralNetwork(
        input_size=state_size, 
        hidden_sizes=cfg.agent.hidden_network_size, 
        output_size=action_size
    )
    critic_network = NeuralNetwork(
        input_size=state_size, 
        hidden_sizes=cfg.agent.hidden_network_size, 
        output_size=1
    )

    agent = A2CAgent(
        state_size = state_size,
        action_size = action_size,
        policy_network=policy_network, 
        critic_network=critic_network,
        device=cfg.device,
        learning_rate=cfg.agent.learning_rate,
        grad_clip=cfg.agent.grad_clip,
        value_loss_coef=cfg.agent.value_loss_coef,
        entropy_coef=cfg.agent.entropy_coef,
        optimizer=None, # use default optimizer, ADAM
    )

    print("Starting trainer")
    trainer = Trainer(
        env=env, 
        buffer=buffer, 
        agent=agent, 
        logger_config=cfg.logger.logger_config, 
        batch_size=cfg.training.batch_size, 
        eval_freq=cfg.training.eval_freq, 
        model_update_freq=cfg.training.model_update_freq, 
        n_update_steps=cfg.training.n_update_steps, 
        model_save_freq=cfg.training.model_save_freq, 
        model_save_path=cfg.training.model_save_path, 
        save_best_model=cfg.training.save_best_model, 
        log_env_info=cfg.training.log_env_info,
        render_evals=cfg.training.render_evals,
    )

    trainer.train(100_000)


if __name__ == "__main__":
    # Set WANDB_MODE to "disabled" to disable wandb logging if needed
    # os.environ["WANDB_MODE"] = "disabled"

    wandb.login()

    env_params = EnvConfig(
            env_name = "GridEnv",
            env_size = 5,
            max_steps_per_episode = 50,
            render_mode = "rgb_array",
    )

    agent_params = AgentConfig(
        model = "A2C",
        policy_network = "NeuralNetwork",
        critic_network = "NeuralNetwork",
        hidden_network_size = [32, 32],
        learning_rate = 0.0005,
        gamma = 0.99,
        grad_clip = 1.0,
        value_loss_coef = 0.5,
        entropy_coef = 0.05,
        optimizer = "ADAM",
    )

    batch_size = 256
    # reccomended to have both batch_size in buffer and training_params be the same for on-policy agents

    buffer_params = BufferConfig(
        type = "OnPolicy",
        name = "TorchTensorBuffer",
        buffer_size = 256,
        batch_size = batch_size,
        gae_lambda= 0.95,
        gamma= 0.99,
    )

    training_params = TrainingParams(
        batch_size = batch_size,
        eval_freq = 100,
        model_update_freq = 1000,
        n_update_steps = 1,
        model_save_freq = 1000,
        model_save_path = './src/trained_agents/',
        save_best_model = True,
        log_env_info = False,
        render_evals = True,
    )
    wandb_config = training_params.to_wandb_config()
    logger_params = LoggerConfig(
        project = "SoccerAI",
        name = "5x5Grid_A2C_100kSteps",
        reinit = False,
        wandb_config = wandb_config,
    )

    config = Config(
        device="cuda" if torch.cuda.is_available() else "mps",
        env=env_params,
        logger=logger_params,
        agent=agent_params,
        training=training_params,
        buffer=buffer_params,
    )

    run(config)
    wandb.finish()