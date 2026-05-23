import wandb
import torch
import os 
from types import SimpleNamespace

from src.utils.configs_to_wandb import ConfigsToWandb
from src.utils.config import Config, SoccerEnvConfig, LoggerConfig, AgentConfig, TrainingParams, BufferConfig
from src.utils.run import run
# python3 -m src.examples.1v0_env_sweep

def train_sweep():
    wandb.init(project="SoccerAI", reinit=True)
    run_name = f"1v1_MARL_A2C_Sweep_Trial_{wandb.run.name}"
    print("Run Name:", run_name)
    
    sweep_config = wandb.config

    agent_params = AgentConfig(
        on_policy = True,
        model = "A2C",
        policy_network = "NeuralNetwork",
        critic_network = "NeuralNetwork",
        transfer_learning = False, 
        model_load_path = "src/trained_agents/optimal_policies/10x10Grid_A2C_250kSteps",
        hidden_network_size = sweep_config.hidden_network_size,
        learning_rate = sweep_config.learning_rate,
        gamma = sweep_config.gamma,
        grad_clip = sweep_config.grad_clip,
        value_loss_coef = sweep_config.value_loss_coef,
        entropy_coef = sweep_config.entropy_coef,
        optimizer = "ADAM",
    )
    
    batch_size = sweep_config.batch_size 

    buffer_params = BufferConfig(
        type = "OnPolicy",
        name = "TorchTensorBuffer",
        buffer_size = batch_size,
        batch_size = batch_size,
        gae_lambda= sweep_config.gae_lambda,
        gamma= sweep_config.gamma, 
    )

    training_params = TrainingParams(
        total_training_steps = 100_000,
        batch_size = batch_size,
        eval_freq = 1_000,
        n_update_steps = 1,
        model_save_freq = 10_000,
        model_save_path = f'./src/trained_agents/1v0_Soccer/{run_name}',
        save_best_model = False,
        log_env_info = False,
        render_evals = True,
    )
    
    env_params = SoccerEnvConfig(
        env_name = "SoccerEnv",
        team_a_size = 1,
        team_b_size = 1,
        width = 100.0,
        height = 60.0,
        time_step = 0.1,
        goal_size = 20.0,
        kf = 20.0,
        fric = 0.85,
        bmw = 0.5,
        pmw = 0.2,
        render_mode= "rgb_array", 
    )
    
    wandb_config = ConfigsToWandb(env_params, agent_params, buffer_params, training_params)
    logger_params = LoggerConfig(
        logger="WandBLogger",
        project = "SoccerAI",
        name=run_name,
        reinit = True, 
        wandb_config = wandb_config, 
        sweep = True, 
        logger_save_freq = 10
    )

    config = Config(
        device="cuda" if torch.cuda.is_available() else "mps",
        env=env_params,
        logger=logger_params,
        agent=agent_params,
        training=training_params,
        buffer=buffer_params,
    )

    # Optional dynamic fields consumed by src/utils/run.py when MARL=True
    # This defines MARLTrainer-wide behavior and two MARL agent slots for 1v1 self-play.
    config.trainer = SimpleNamespace(
        shared_buffer=True,
        equal_batch_size=True,
        update_same_time=True,
        batch_size=batch_size,
        model_update_freq=training_params.model_update_freq,
        n_update_steps=training_params.n_update_steps,
    )
    config.marl_agent_configs = {
        "team_a_agent": {
            "is_on_policy": True,
            "batch_size": batch_size,
            "n_epochs": training_params.n_epochs,
        },
        "team_b_agent": {
            "is_on_policy": True,
            "batch_size": batch_size,
            "n_epochs": training_params.n_epochs,
        },
    }

    print("Starting Run")

    run(config, num_post_eval_runs=10, print_config=True, MARL=True)
        


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled" 
    wandb.login()

    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'name': 'final/ep_rewards', 
            'goal': 'maximize'
            },
        'parameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.005},
            'gae_lambda': {'min': 0.8, 'max': 1.0},
            'gamma': {'values': [0.99, 0.995, 0.999]},
            'value_loss_coef': {'min': 0.25, 'max': 0.75},
            'grad_clip': {'min': 0.5, 'max': 2.0},
            'hidden_network_size': {'values': [[32, 32], [64, 64], [16, 16]]},
            'entropy_coef': {'min': 0.001, 'max': 0.1},
            'batch_size': {'values': [32, 64, 128, 256, 512]}
            }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project="SoccerAI"
    )

    wandb.agent(sweep_id, function=train_sweep, count=10)