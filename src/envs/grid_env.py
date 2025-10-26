import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
import time
import matplotlib.pyplot as plt

from mod_base_gym_env import modBaseGymEnv

class GridEnv(modBaseGymEnv):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 5, terminating_step=200, render_mode: str = None) -> None:
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        self.agent_location = None 
        self.target_location = None

        self.grid_size = 60 # Pixels per grid cell
        self.screen_dim = self.size * self.grid_size

        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=torch.int32), 
        #         "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=torch.int32),
        #     }
        # )
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=size-1, 
            shape=(4,), 
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(4)

        self.map_actions_to_movement = {
            0: torch.tensor([1, 0], dtype=torch.int32), # right
            1: torch.tensor([-1, 0], dtype=torch.int32), # left
            2: torch.tensor([0, 1], dtype=torch.int32), # up
            3: torch.tensor([0, -1], dtype=torch.int32), # down
        }

        # dist function for reward generation
        self.dist_fn = nn.PairwiseDistance(p=2)

        self.timestep = -1
        self.terminating_step = terminating_step

    def change_render_mode(self, new_render_mode):
        self.render_mode = new_render_mode

    def get_size(self):
        return self.size
    
    def _get_state(self):
        # print(self.agent_location, self.target_location)
        state_tensor = torch.cat((self.agent_location, self.target_location))
        return state_tensor.float()
        # return {"agent": self.agent_location, "target": self.target_location}
    
    def reset(self, seed: int = None, options=None):
        super().reset(seed=seed)
        
        self.agent_location = torch.randint(0, self.size, (2,), dtype=torch.int32)
        while True:
            self.target_location = torch.randint(0, self.size, (2,), dtype=torch.int32)
            if not torch.equal(self.target_location, self.agent_location):
                break

        state = self._get_state()
        info = {}
        
        self.timestep = 0
        
        return state, info

    def step(self, action: int):
        self.timestep += 1

        move_direction = self.map_actions_to_movement[action]

        self.agent_location = torch.clip(
            self.agent_location + move_direction, 
            min=0, 
            max=self.size-1,
        )
        terminated = bool(torch.equal(self.agent_location, self.target_location))
        
        if terminated:
            reward = 1
        else:
            reward = -.1 * self.dist_fn(self.agent_location.float(), self.target_location.float()).item()

        truncated = self.timestep >= self.terminating_step
        
        state = self._get_state()
        info = {}
        
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            return state, reward, terminated, truncated, info, self.render()
        return state, reward, terminated, truncated, info
    
    def render(self):
        # double check render mode
        if self.render_mode == "human": 
            return self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_frame()
        
    def _render_frame(self):
        os.system('cls' if os.name == 'nt' else 'clear')

        agent_pos = self.agent_location.numpy().astype(int)
        target_pos = self.target_location.numpy().astype(int)

        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        grid[target_pos[1]][target_pos[0]] = 'T'

        grid[agent_pos[1]][agent_pos[0]] = 'A'

        output = f"--- GridEnv (Size: {self.size}x{self.size}) ---\n"
        output += f"Timestep: {self.timestep} / {self.terminating_step}\n"
        output += f"Agent (A): ({agent_pos[0]}, {agent_pos[1]}), Target (T): ({target_pos[0]}, {target_pos[1]})\n"
        
        output += "+" + "--" * self.size + "+\n"
        
        for row in reversed(grid): 
            output += "|" + " ".join(row) + " |\n"
        
        output += "+" + "--" * self.size + "+\n"

        print(output)
    
    def _render_rgb_frame(self):
        # create empty background
        canvas = np.zeros((self.screen_dim, self.screen_dim, 3), dtype=np.uint8) + 255 

        agent_pos = self.agent_location.numpy().astype(int)
        target_pos = self.target_location.numpy().astype(int)

        target_x_start = target_pos[0] * self.grid_size
        target_y_start = (self.size - 1 - target_pos[1]) * self.grid_size
        
        canvas[
            target_y_start : target_y_start + self.grid_size,
            target_x_start : target_x_start + self.grid_size,
        ] = [50, 168, 82] 

        agent_x_start = agent_pos[0] * self.grid_size
        agent_y_start = (self.size - 1 - agent_pos[1]) * self.grid_size

        canvas[
            agent_y_start : agent_y_start + self.grid_size,
            agent_x_start : agent_x_start + self.grid_size,
        ] = [252, 186, 3] 

        return canvas

    def close(self):
        # No cleanup needed
        pass

if __name__ == "__main__":
    test_env = True
    register = True
    render_mode = "rgb_array"

    if register:
        # Register gymnasium environment
        gym.register(
            id="GridEnv-v0",
            entry_point=GridEnv,
            max_episode_steps=200, 
        )

    if test_env: 
        if render_mode == "human":
            print("Render mode: human, text-based rendering")
            # Create an environment with render_mode="human"
            env = GridEnv(size=8, render_mode="human", terminating_step=15)

            state, info = env.reset()
            done = False        
            while not done:            
                action = env.action_space.sample()

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                print(f"Action: {['Right', 'Left', 'Up', 'Down'][action]}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
                time.sleep(0.5) # pause to create FPS

            env.close()
        
        elif render_mode == "rgb_array":
            print("Render mode: rgb_array, visual rendering")
            # Create an environment with render_mode="rgb_array"
            env = GridEnv(size=8, render_mode="rgb_array", terminating_step=15)

            state, info = env.reset()
            done = False        
            while not done:            
                action = env.action_space.sample()

                state, reward, terminated, truncated, info, frame = env.step(action)
                done = terminated or truncated
                
                print(f"Action: {['Right', 'Left', 'Up', 'Down'][action]}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
                
                # Display the frame using an external library like OpenCV or matplotlib
                # Here, we'll use matplotlib for simplicity
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.5) # pause to create FPS
                plt.clf()

            env.close()