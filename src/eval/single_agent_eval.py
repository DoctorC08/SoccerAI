import numpy as np
import wandb
import os

from src.eval.base_eval import BaseEval
from src.envs.mod_base_gym_env import modBaseGymEnv

class SingleAgentEval(BaseEval): 
    def __init__(self, render_evals: bool, env: modBaseGymEnv, get_action, get_metrics) -> None:
            super().__init__(
                render_evals=render_evals, 
                env=env, 
                get_action=get_action, 
                get_metrics=get_metrics
            )

    def eval(self, log=True):
        if self.render_evals:
            self.env.change_render_mode('rgb_array')
            state, _ = self.env.reset()
            eval_ep_rews = 0
            length = 0

            eval_renderings = []

            while True:
                action, logits = self.get_action(state, is_training=False)
                if self.render_evals: 
                    state, reward, terminated, truncated, _, render = self.env.step(action)
                else:
                    state, reward, terminated, truncated, _ = self.env.step(action)
                eval_ep_rews += reward
                eval_metrics = self.get_metrics(logits, logger_dir="eval/")
                length += 1

                if self.render_evals:
                    # Append render to eval_renderings shape: (t, height, width, channels)
                    eval_renderings.append(render)

                if terminated or truncated:
                    break
            

            # Reshape eval_renderings to (t, channels, height, width)
            if log: 
                eval_renderings = np.array(eval_renderings)
                if eval_renderings.ndim == 3:
                    # if only single frame, expand dim
                    eval_renderings = np.expand_dims(eval_renderings, axis=0)

                elif eval_renderings.ndim == 2:
                    # If only single grayscale frame, expand dims
                    eval_renderings = np.expand_dims(eval_renderings, axis=-1)
                    eval_renderings = np.expand_dims(eval_renderings, axis=0)
                if eval_renderings.ndim == 4:
                    eval_render_T = np.transpose(eval_renderings, (0, 3, 1, 2))
                    wandb_video = wandb.Video(eval_render_T, 
                                        fps=self.fps, 
                                        format="mp4", 
                                        caption=f"{self.env.__class__.__name__} Render: Eval at episode: {self.n_eps}, \
                                            rew: {eval_ep_rews}")
                    self.logger.log({"eval/video": wandb_video}, self.n_eps)
                elif eval_renderings.ndim == 1:
                    # if no video collected
                    pass
                else: 
                    print(f"Error: Final rendering array has unexpected dimensions: {eval_renderings.ndim}")

                self.logger.log({
                        "eval/ep_rewards": eval_ep_rews, 
                        "eval/length": length
                    } | eval_metrics, self.n_eps) 
            
            if self.save_best_model:
                self.cur_score = (self.best_model_exp_moving_avg * self.cur_score) + \
                                ((1 - self.best_model_exp_moving_avg) * eval_ep_rews)
                if self.cur_score > self.save_threshold: 
                    if os.path.exists(self.model_save_path) is False:
                        os.makedirs(self.model_save_path)
                    self.save_agents(path_name=f"{self.model_save_path}/" + self.name)
                    self.save_threshold = self.cur_score
            return {
                "final/ep_rewards": eval_ep_rews, 
                "final/length": length
            } 