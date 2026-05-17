from src.envs.mod_base_gym_env import modBaseGymEnv
from src.envs.soccer_envs.soccer_renderer import SoccerRenderer
import gymnasium as gym
from src.envs.soccer_envs import soccer_sim
import torch
import numpy as np

class SoccerEnv(modBaseGymEnv):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 4}
    def __init__(self, 
                 team_a_size: int = 2, 
                 team_b_size: int = 2, 
                 width: float = 100.0, 
                 height: float = 60.0, 
                 time_step: float = 0.1, 
                 goal_size: float = 20, 
                 kf: float = 20.0, 
                 fric: float = 0.85, 
                 bmw: float = 0.5, 
                 pmw: float = 0.2, 
                 max_steps: int = 1000,
                 random_ball_placement: bool = False,
                 render_mode: str = None, 
                 **kwargs) -> None:
        '''
        Custom Soccer Environment for Multi-Agent Reinforcement Learning.
        Initialize the Soccer Environment.
        Args:
            team_a_size (int): Number of players in Team A.
            team_b_size (int): Number of players in Team B.
            width (float): Width of the soccer field.
            height (float): Height of the soccer field.
            time_step (float): Time step for the simulation.
            goal_size (float): Size of the goals.
            kf (float): Kick force applied when a player kicks the ball.
            fric (float): Friction coefficient affecting player and ball movement.
            bmw (float): Ball mass weight, affecting how the ball responds to kicks and collisions.
            pmw (float): Player mass weight, affecting how players respond to collisions and movement.
            random_ball_placement (bool): If True, place the ball randomly inside the middle region on reset.
            render_mode (str): Rendering mode for the environment ("human", "rgb_array", or None).
        '''
        self.team_a_size = team_a_size
        self.team_b_size = team_b_size

        # if human render mode force team size to be 1v0
        if render_mode == "human":
            self.team_a_size = 1
            self.team_b_size = 0

        self.num_players = self.team_a_size + self.team_b_size
        self.width = width
        self.height = height
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_step = 0
        self.last_actions = []
        self.action_history = []
        self.render_mode = render_mode
        self.goal_size = goal_size
        self.env = soccer_sim.SoccerEnv(
            num_a = self.team_a_size,
            num_b = self.team_b_size,
            w = self.width,
            h = self.height,
            time_step = self.time_step,
            g_size = self.goal_size,
            kf=kf,
            fric=fric,
            bmw=bmw,
            pmw=pmw,
            max_steps=max_steps,
            random_ball_spawn=random_ball_placement,
            **kwargs,
        )
        self.action_space = gym.spaces.Discrete(6) 

        self.next_action = None 

        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(4 + self.num_players * 4,),
            dtype=np.float32,
        )
        if render_mode in ["human", "rgb_array"]:
            self.renderer = SoccerRenderer(h=600, w=1000, ui_width=400, display_mode=render_mode)
        
    def change_render_mode(self, new_render_mode):
        self.render_mode = new_render_mode
        self.next_action = None

        if new_render_mode in ["human", "rgb_array"]:
            if hasattr(self, "renderer"):
                self.renderer.display_mode = new_render_mode
            else:
                self.renderer = SoccerRenderer(h=600, w=1000, ui_width=400, display_mode=new_render_mode)


    def step(self, actions):
        if self.next_action is not None:
            actions = self.next_action

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().tolist()
        elif isinstance(actions, np.ndarray):
            actions = actions.tolist()

        if isinstance(actions, (int, np.integer)):
            actions = [int(actions)]
        elif isinstance(actions, list):
            actions = [int(a) for a in actions]
        elif isinstance(actions, tuple):
            actions = [int(a) for a in actions]
        else:
            raise TypeError(f"Unsupported action type for SoccerEnv.step: {type(actions)}")

        self.last_actions = list(actions)
        self.action_history.append((self.current_step + 1, list(actions)))
        if len(self.action_history) > 4:
            self.action_history = self.action_history[-8:]

        # print(f"[SoccerEnv.step] Actions sent to C++ backend: {actions}")
        self.env.step(actions)
        self.current_step += 1
        state = self.env.get_state()
        reward = self.env.get_rewards()
        terminated = self.env.is_done() 
        
        if self.render_mode == "human":
            self.next_action = self.render()
        if self.render_mode == "rgb_array":
            return torch.tensor(state), reward, terminated, False, {}, self.render()
        return torch.tensor(state), reward, terminated, False, {}

    def reset(self):
        self.env.reset()
        self.next_action = None
        self.current_step = 0
        self.last_actions = []
        self.action_history = []
        return torch.tensor(self.env.get_state()), {}
    
    def render(self):
        if not hasattr(self, "renderer") and self.render_mode in ["human", "rgb_array"]:
            self.renderer = SoccerRenderer(h=600, w=1000, ui_width=400, display_mode=self.render_mode)
        if hasattr(self, "renderer"):
            self.renderer.display_mode = self.render_mode

        remaining_steps = max(0, self.max_steps - self.current_step)
        timer_info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "remaining_steps": remaining_steps,
            "remaining_seconds": remaining_steps * self.time_step,
        }
        action_info = {
            "last_actions": self.last_actions,
            "history": self.action_history,
        }

        if self.render_mode == "human":
            action = self.renderer.draw(
                self.env.get_state(),
                self.env.get_rewards(),
                timer_info=timer_info,
                action_info=action_info,
            )
            return action
        return self.renderer.draw(
            self.env.get_state(),
            self.env.get_rewards(),
            timer_info=timer_info,
            action_info=action_info,
        )

    def close(self):
        pass 

if __name__ == '__main__':
    env = SoccerEnv(team_a_size=1, team_b_size=0, max_steps=150, render_mode="human")
    obs, info = env.reset()
    action = [0]
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs, reward, terminated)
        if terminated:
            break