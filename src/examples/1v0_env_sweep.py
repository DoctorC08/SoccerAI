import wandb
import torch
import os 

from src.utils.configs_to_wandb import ConfigsToWandb
from src.utils.config import Config, SoccerEnvConfig, LoggerConfig, AgentConfig, TrainingParams, BufferConfig, TrainerConfig
from src.utils.run import run
# python3 -m src.examples.1v0_env_sweep

def train_sweep():
    wandb.init(project="SoccerAI", reinit=True)
    run_name = f"1v0_MARL_A2C_Sweep_Trial_{wandb.run.name}"
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
        buffer_size = batch_size, #TODO: Does batch size and buffer size need to be the same...?
        batch_size = batch_size,
        gae_lambda= sweep_config.gae_lambda,
        gamma= sweep_config.gamma, 
    )

    training_params = TrainingParams(
        total_training_steps = 100_000,
        batch_size = batch_size,
        eval_freq = 2_000,
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
        team_b_size = 0,
        width = 50.0,
        height = 30.0,
        time_step = 0.1,
        goal_size = 15.0,
        kf = 20.0,
        fric = 0.85,
        bmw = 0.5,
        pmw = 0.2,
        max_steps = 150,
        render_mode= "rgb_array", 
    )

    trainer_params = TrainerConfig(
        shared_buffer=True,
        equal_batch_size=True,
        update_same_time=True,
        batch_size=batch_size,
        model_update_freq=training_params.model_update_freq,
        n_update_steps=training_params.n_update_steps,
    )
    
    wandb_config = ConfigsToWandb(env_params, agent_params, buffer_params, training_params, trainer_params)
    logger_params = LoggerConfig(
        logger="WandBLogger",
        project = "SoccerAI",
        name=run_name,
        reinit = True, 
        wandb_config = wandb_config, 
        sweep = True, 
        logger_save_freq = 1
    )

    config = Config(
        device="cuda" if torch.cuda.is_available() else "mps",
        env=env_params,
        logger=logger_params,
        agent=agent_params,
        training=training_params,
        buffer=buffer_params,
        trainer=trainer_params,
    )
    config.marl_agent_configs = {
        "team_a_agent": {
            "is_on_policy": True,
            "batch_size": batch_size,
            "n_epochs": training_params.n_epochs,
        },
    }

    print("Starting Run")

    run(config, num_post_eval_runs=10, print_config=True, MARL=True)
        


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled" 
    wandb.login()

    sweep_configuration = {
        'method': 'random',
        'metric': {
            'name': 'final/ep_rewards', 
            'goal': 'maximize'
            },
        'parameters': {
            'learning_rate': {'values': [0.0001, 0.0003, 0.0007, 0.001]},
            'gae_lambda': {'values': [0.92, 0.95, 0.97]},
            'gamma': {'values': [0.99, 0.995]},
            'value_loss_coef': {'values': [0.4, 0.5, 0.6]},
            'grad_clip': {'values': [0.5, 1.0]},
            'hidden_network_size': {'values': [[64, 64], [128, 128]]},
            'entropy_coef': {'values': [0.005, 0.01, 0.02]},
            'batch_size': {'values': [128, 256]}
            }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project="SoccerAI"
    )

    wandb.agent(sweep_id, function=train_sweep, count=20)