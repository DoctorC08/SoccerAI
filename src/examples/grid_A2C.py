import torch
import wandb
import os

from src.eval.single_agent_eval import SingleAgentEval
from src.utils.configs_to_wandb import ConfigsToWandb
from src.utils.config import Config, GridEnvConfig, LoggerConfig, AgentConfig, TrainingParams, BufferConfig, EvalParams
from src.utils.run import run

if __name__ == "__main__":
    # Set WANDB_MODE to "disabled" to disable wandb logging if needed
    # os.environ["WANDB_MODE"] = "disabled"

    wandb.login()

    env_params = GridEnvConfig(
            env_name = "GridEnv",
            env_size = 10,
            max_steps_per_episode = 50,
            render_mode = "rgb_array",
    )

    agent_params = AgentConfig(
        on_policy = True,
        model = "A2C",
        policy_network = "NeuralNetwork",
        critic_network = "NeuralNetwork",
        transfer_learning = False, 
        model_load_path = "src/trained_agents/optimal_policies/5x5Grid_A2C_100kSteps",
        hidden_network_size = [32, 32],
        learning_rate = 0.0005,
        gamma = 0.99,
        grad_clip = 1.0,
        value_loss_coef = 0.5,
        entropy_coef = 0.05,
        optimizer = "ADAM",
        batch_size=256, 
    )

    # recommended to have both batch_size in buffer and training_params be the same for on-policy agents
    batch_size = 256

    buffer_params = BufferConfig(
        type = "OnPolicy",
        name = "TorchTensorBuffer",
        buffer_size = 256,
        batch_size = batch_size,
        gae_lambda= 0.95,
        gamma= 0.99,
    )

    training_params = TrainingParams(
        total_training_steps = 1_000_000,
        eval_freq = 500,
        model_save_freq = 10_000,
        model_save_path = './src/trained_agents/10x10_Grid_A2C',
        save_best_model = True,
        log_env_info = False,
        render_evals = True,
    )
    wandb_config = ConfigsToWandb(env_params, agent_params, buffer_params, training_params)
    logger_params = LoggerConfig(
        logger="WandBLogger",
        project = "SoccerAI",
        name = f"{env_params.env_size}x{env_params.env_size}{env_params.env_name}_{agent_params.model}_1mSteps",
        reinit = False,
        wandb_config = wandb_config,
    )

    evaluator = EvalParams(evaluator=SingleAgentEval)

    # print(wandb_config)

    config = Config(
        device="cuda" if torch.cuda.is_available() else "mps",
        env=env_params,
        evaluator=evaluator,
        logger=logger_params,
        agent=agent_params,
        training=training_params,
        buffer=buffer_params,
    )

    run(config, num_post_eval_runs=10)
    wandb.finish()