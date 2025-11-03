from src.agents.neural_network import NeuralNetwork
from src.agents.on_policy_agents.A2C import A2CAgent
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer
from src.envs.grid_env import GridEnv
from src.train.trainer import Trainer
from src.config import Config 


def run(cfg: Config):
    if cfg.env.env_name == "GridEnv":
        env = GridEnv(
            size = cfg.env.env_size, 
            terminating_step= cfg.env.max_steps_per_episode,
            render_mode = cfg.env.render_mode,
        )
    else: 
        raise NotImplementedError(f"Environment {cfg.env.env_name} not implemented in this example.")

    if cfg.buffer.type == "OnPolicy":
        if cfg.buffer.name == "TorchTensorBuffer":
            buffer = TorchTensorBuffer(
                max_size=cfg.buffer.buffer_size,
                device=cfg.device,
                batch_size=cfg.buffer.batch_size,
                gamma=cfg.buffer.gamma,
                gae_lambda=cfg.buffer.gae_lambda
            )
        else:
            raise NotImplementedError(f"Buffer {cfg.buffer.name} not implemented in this example.")
    else:
        raise NotImplementedError(f"Buffer type {cfg.buffer.type} not implemented in this example.")

    state_size=env.observation_space.shape[0], 
    action_size=env.action_space.n,

    if cfg.agent.on_policy: 
        if cfg.agent.policy_network == "NeuralNetwork":
            policy_network = NeuralNetwork(
                input_size=state_size, 
                hidden_sizes=cfg.agent.hidden_network_size, 
                output_size=action_size
            )
        if cfg.agent.critic_network == "NeuralNetwork":
            critic_network = NeuralNetwork(
                input_size=state_size, 
                hidden_sizes=cfg.agent.hidden_network_size, 
                output_size=1
            )
        else:
            raise NotImplementedError(f"Policy or Critic network {cfg.agent.policy_network} not implemented in this example.")
    else:
        raise NotImplementedError("Only on-policy agents are implemented in this example.")

    if cfg.agent.model == "A2C":
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
    else:
        raise NotImplementedError(f"Agent model {cfg.agent.model} not implemented in this example.")

    if cfg.agent.transfer_learning:
        assert cfg.agent.model_load_path != "", "Model load path must be specified for transfer learning."
        agent.load_model(cfg.agent.model_load_path)

    trainer = Trainer(
        agent=agent, 
        buffer=buffer, 
        env=env,
        logger_config=cfg.logger.logger_config, 
        logger=cfg.logger.logger,
        batch_size=cfg.training.batch_size, 
        eval_freq=cfg.training.eval_freq, 
        model_update_freq=cfg.training.model_update_freq, 
        n_update_steps=cfg.training.n_update_steps, 
        n_epochs=cfg.training.n_epochs,
        model_save_freq=cfg.training.model_save_freq, 
        model_save_path=cfg.training.model_save_path, 
        save_best_model=cfg.training.save_best_model, 
        best_model_exp_moving_avg=cfg.training.best_model_exp_moving_avg,
        log_env_info=cfg.training.log_env_info,
        env_info_fn=cfg.training.env_info_fn,
        render_evals=cfg.training.render_evals,
        fps=cfg.training.fps,
    )

    trainer.train(cfg.training.total_training_steps)