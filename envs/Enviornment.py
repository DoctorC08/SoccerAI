# Import the necessary libraries
import pygame
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
# Verbose
global verbose
verbose = False

class GridSoccer(gym.Env):
    def __init__(self, render_mode=None, size=10):
        self.render_mode = render_mode

        self.timestep = 0 # timesteps
        self.size = size # Size of grid
        self.window_size = 512 # PyGame window size
        self.max_timesteps = 50

        self.goal_loc = np.array([size - 1, size // 2])

        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(4,), dtype=int)

        # (0) Up (1) Down (2) Left (3) Right
        self.action_space = spaces.Discrete(4)


        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }



        # if human mode (Viewing) is used then window and clock will be needed
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.has_ball == 0:
            return np.concatenate((self._agent_location, self._target_location))
        else:
            return np.append(self._agent_location, self.goal_loc)

    def _get_info(self): # return manhattan distance
        return np.linalg.norm(self._agent_location - self._target_location, ord=1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set agent location
        self._agent_location = np.array([0, self.size // 2])

        self.truncated = False
        self.timestep = 0
        self.reward = 0

        self.has_ball = 0
        self.first_time = 0

        self._target_location = self._agent_location
        while (np.array_equal(self._target_location, self._agent_location)
               or np.array_equal(self._target_location, self.goal_loc)):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.timestep += 1
        action = int(action)
        direction = self._action_to_direction[action]

        # clip to make sure agent doesn't leave grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self.goal_loc) and self.has_ball == 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self.reward = 0
        if terminated:
            self.reward = 1 - ((self.timestep / self.max_timesteps) / 10)
        elif (np.array_equal(self._target_location, self._agent_location)
              and self.first_time == 0):
            self.first_time = 1
            self.has_ball = 1
            self.reward = 0.5 - ((self.timestep / self.max_timesteps) / 10)

        elif self.timestep > self.max_timesteps:
            self.reward = -1
            self.truncated = True
        # print(self.timestep, self.reward, terminated, self.truncated, self._agent_location, self._target_location)

        return observation, self.reward, terminated, self.truncated, info

    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame()

    def _render_frame(self):

        # if self.render_mode == "human":
        #     print("render mode is human")
        # else:
        #     print("render mode is", self.render_mode)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Then we draw the goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self.goal_loc,
                (pix_square_size, pix_square_size)
            )
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(10)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
