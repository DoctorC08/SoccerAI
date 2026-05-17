from src.agents.neural_network import NeuralNetwork
from src.agents.on_policy_agents.A2C import A2CAgent
from src.buffers.on_policy_buffers.torch_tensor_buffer import TorchTensorBuffer

from src.envs.grid_env import GridEnv
from src.envs.soccer_envs.soccer_env import SoccerEnv

from src.train.trainer import Trainer
from src.train.MARL_trainer import MARLTrainer
from src.utils.config import Config 


def run(cfg: Config, run_eval: int = 0, print_config=False, MARL=False) -> dict | None:
    '''
    Run training with the given configuration.
    Args:
        cfg (Config): Configuration object containing all parameters.
        run_eval (int): (0 will skip evaluation) Number of evaluation episodes to run after training. 
    Returns:
        dict | None: Evaluation metrics if run_eval is True, else None.
    '''
    if MARL: 
        if cfg.env.env_name == "SoccerEnv":
            env = SoccerEnv(
                team_a_size=cfg.env.team_a_size,
                team_b_size=cfg.env.team_b_size,
                width=cfg.env.width,
                height=cfg.env.height,
                time_step=cfg.env.time_step,
                goal_size=cfg.env.goal_size,
                kf=cfg.env.kf,
                fric=cfg.env.fric,
                bmw=cfg.env.bmw,
                pmw=cfg.env.pmw,
                max_steps=cfg.env.max_steps,
                random_ball_placement=cfg.env.random_ball_placement,
                render_mode = cfg.env.render_mode,
                **cfg.env.sim_kwargs,
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

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

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

        print("Initializing Agents")

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

        print("Initializing Trainer")

        def _make_marl_buffer() -> TorchTensorBuffer:
            return TorchTensorBuffer(
                max_size=cfg.buffer.buffer_size,
                device=cfg.device,
                batch_size=cfg.buffer.batch_size,
                gamma=cfg.buffer.gamma,
                gae_lambda=cfg.buffer.gae_lambda,
            )

        trainer_cfg = getattr(cfg, "trainer", None)
        shared_buffer = getattr(trainer_cfg, "shared_buffer", True)
        equal_batch_size = getattr(trainer_cfg, "equal_batch_size", True)
        update_same_time = getattr(trainer_cfg, "update_same_time", True)
        marl_batch_size = getattr(trainer_cfg, "batch_size", cfg.training.batch_size)
        marl_model_update_freq = getattr(trainer_cfg, "model_update_freq", cfg.training.model_update_freq)
        marl_n_update_steps = getattr(trainer_cfg, "n_update_steps", cfg.training.n_update_steps)

        if not shared_buffer and equal_batch_size:
            print("Warning: equal_batch_size requires shared_buffer=True in current MARLTrainer implementation. Setting equal_batch_size=False.")
            equal_batch_size = False

        user_agent_configs = getattr(cfg, "marl_agent_configs", None)
        marl_agent_configs = {}

        if user_agent_configs is None:
            agent_buffer = buffer if shared_buffer else _make_marl_buffer()
            if cfg.agent.on_policy:
                marl_agent_configs["agent_0"] = (
                    agent,
                    agent_buffer,
                    True,
                    marl_batch_size,
                    cfg.training.n_epochs,
                )
            else:
                marl_agent_configs["agent_0"] = (
                    agent,
                    agent_buffer,
                    False,
                    marl_batch_size,
                    marl_model_update_freq,
                    marl_n_update_steps,
                )
        else:
            if isinstance(user_agent_configs, dict):
                iterable_configs = user_agent_configs.items()
            else:
                iterable_configs = enumerate(user_agent_configs)

            for idx, data in iterable_configs:
                key = idx if isinstance(idx, str) else f"agent_{idx}"
                if not isinstance(data, dict):
                    raise TypeError("Each item in cfg.marl_agent_configs must be a dict.")

                agent_obj = data.get("agent", agent)
                is_on_policy = data.get("is_on_policy", cfg.agent.on_policy)
                agent_batch_size = data.get("batch_size", marl_batch_size)

                if "buffer" in data:
                    agent_buffer = data["buffer"]
                else:
                    agent_buffer = buffer if shared_buffer else _make_marl_buffer()

                if is_on_policy:
                    n_epochs = data.get("n_epochs", cfg.training.n_epochs)
                    marl_agent_configs[key] = (
                        agent_obj,
                        agent_buffer,
                        True,
                        agent_batch_size,
                        n_epochs,
                    )
                else:
                    agent_update_freq = data.get("model_update_freq", marl_model_update_freq)
                    agent_n_update_steps = data.get("n_update_steps", marl_n_update_steps)
                    marl_agent_configs[key] = (
                        agent_obj,
                        agent_buffer,
                        False,
                        agent_batch_size,
                        agent_update_freq,
                        agent_n_update_steps,
                    )

        trainer = MARLTrainer(
            agents=marl_agent_configs,
            env=env,
            logger_config=cfg.logger.logger_config, 
            logger=cfg.logger.logger,
            logger_save_freq=cfg.logger.logger_save_freq,
            eval_freq=cfg.training.eval_freq, 
            model_save_freq=cfg.training.model_save_freq, 
            model_save_path=cfg.training.model_save_path, 
            save_best_model=cfg.training.save_best_model, 
            best_model_exp_moving_avg=cfg.training.best_model_exp_moving_avg,
            log_env_info=cfg.training.log_env_info,
            env_info_fn=cfg.training.env_info_fn,
            render_evals=cfg.training.render_evals,
            fps=cfg.training.fps,
            shared_buffer=shared_buffer,
            buffer=buffer if shared_buffer else None,
            equal_batch_size=equal_batch_size,
            batch_size=marl_batch_size,
            update_same_time=update_same_time,
            model_update_freq=marl_model_update_freq,
            n_update_steps=marl_n_update_steps,
        )
        
    else: 
        if cfg.env.env_name == "GridEnv":
            env = GridEnv(
                size = cfg.env.env_size, 
                terminating_step= cfg.env.max_steps_per_episode,
                render_mode = cfg.env.render_mode,
            )
        elif cfg.env.env_name == "SoccerEnv":
            env = SoccerEnv(
                team_a_size=cfg.env.team_a_size,
                team_b_size=cfg.env.team_b_size,
                width=cfg.env.width,
                height=cfg.env.height,
                time_step=cfg.env.time_step,
                goal_size=cfg.env.goal_size,
                kf=cfg.env.kf,
                fric=cfg.env.fric,
                bmw=cfg.env.bmw,
                pmw=cfg.env.pmw,
                random_ball_placement=cfg.env.random_ball_placement,
                render_mode = cfg.env.render_mode,
                **cfg.env.sim_kwargs,
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

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

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

        print("Initializing Agents")

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

        print("Initializing Trainer")

        trainer = Trainer(
            agent=agent, 
            buffer=buffer, 
            env=env,
            logger_config=cfg.logger.logger_config, 
            logger=cfg.logger.logger,
            logger_save_freq=cfg.logger.logger_save_freq,
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

    if print_config:
        print(cfg)

    trainer.train(cfg.training.total_training_steps, run_eval=run_eval)