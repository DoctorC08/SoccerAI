# Import the necessary libraries
import numpy as np

import pygame
import gym
from gym import spaces
from gym.spaces import Box, Tuple, Discrete

import numpy as np
import torch

import copy

import time


# Verbose
global verbose
verbose = False

field_width = 345
field_height = 225
# Set up the drawing window
screen_width_ft = (field_width + 10 + 10) * 2 # 10 feet on each side
screen_height_ft = (field_height + 20 + 20) * 2  # 20 feet on each side

# Calculate the center of the screen
center_x = screen_width_ft // 2
center_y = screen_height_ft // 2

# Define the size of the centered box
box_width = field_width * 2
box_height = field_height * 2


# Calculate the position to center the box
box_x = center_x - (box_width // 2)
box_y = center_y - (box_height // 2)

# Define the Player class for pygame
class Player:
    def __init__(self):
        self.radius = 8  # Radius for the circle
        self.x = center_x
        self.y = center_y
        self.x_velo = 0
        self.y_velo = 0
        self.team = 0
        self.orientation = 0

    def move(self, x_pos, y_pos, velo, orientation, team):
        if verbose:
            print("velo:", velo)
        self.x = float(x_pos)
        self.y = float(y_pos)
        self.x_velo = float(velo[0][0])
        self.y_velo = float(velo[0][1])
        self.orientation = float(orientation)
        self.team = float(team)

    def draw(self, surface):
        # Draw player position
        if self.team == 0:
            pygame.draw.circle(surface, (0, 0, 255), (self.x, self.y), self.radius)
        else:
            pygame.draw.circle(surface, (255, 0, 0), (self.x, self.y), self.radius)

        # Draw player velo and orientation
        pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + self.x_velo, self.y + self.y_velo), width=2) # Velo
        # Orientations:
        pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + ((np.cos(self.orientation - (np.pi / 4))) * self.radius), self.y + ((np.sin(self.orientation - (np.pi / 4))) * self.radius)), width=2)
        pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + ((np.cos(self.orientation + (np.pi / 4))) * self.radius), self.y + ((np.sin(self.orientation + (np.pi / 4))) * self.radius)), width=2)




# Define the ball for pygame
class Ball:
    def __init__(self):
        self.radius = 3
        self.x = center_x
        self.y = center_y
        self.x_velo = 0
        self.y_velo = 0
    def move(self, x_pos, y_pos, velo):
        self.x = float(x_pos)
        self.y = float(y_pos)
        self.x_velo = float(velo[0])
        self.y_velo = float(velo[1])
    def draw(self, surface):
        if verbose:
            print("self.x and self.y", self.x, self.y)

        # Draw ball
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), self.radius)
        # Draw velocity vector
        pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + self.x_velo, self.y + self.y_velo), width=2)

class env(gym.Env):
    def __init__(self):
        super(env, self).__init__()
        self.state = 0  # Initial state

        # Define field widths and heights and bounds
        self.field_width = 345 * 2
        self.field_height = 225 * 2
        self.field_bounds_x = 10 * 2
        self.field_bounds_y = 20 * 2

        # multiplied by 2 for scale divided for half
        self.center_y = (self.field_height + self.field_bounds_y * 2) // 2

        # define goal positions
        self.center_goal_position_x1 = self.field_bounds_x
        self.center_goal_position_x2 = self.field_bounds_x + self.field_width

        # y coordinate is same for both
        self.center_goal_position_y = self.field_height + (self.field_bounds_y / 2)

        # Define number of players (both teams)
        self.num_players = 4
        # Define player speed
        self.player_speed = 5
        # Define max player speed
        self.player_max_speed = 20
        # Define player sprint bar timer
        self.total_sprint_bar = 100
        # Define sprint speed
        self.sprint_speed = self.player_speed * 1.3

        # define passing velo
        self.pass_velo = 20

        # define meu friction
        self.meu_friction = 0.1
        self.ball_weight = 0.45
        self.meu_friction_force = self.meu_friction * self.ball_weight

        # Step function!
        self.step_reward = [0, 0]

        #define timesteps
        self.max_steps = 3000
        self.current_step = 0

        # define action and obs space
        # Define the action space
        self.action_space = Tuple((
            Discrete(4),
            Discrete(3),
            Discrete(2),
            Discrete(3)
        ))
        # Repeat the action space num player times
        self.action_space = Tuple([self.action_space for i in range(self.num_players)])

        # Define obs space
        self.observation_space = Tuple((
            Box(low=self.field_bounds_x, high=self.field_bounds_x + self.field_width, shape=(self.num_players, 2)),  # player_positions
            Box(low=0, high=np.pi * 2, shape=(self.num_players,)),  # player_orientations
            Box(low=0, high=self.player_max_speed, shape=(self.num_players, 2)),  # player_velos
            Box(low=0, high=self.total_sprint_bar, shape=(self.num_players,)),  # player_sprint_bar
            Discrete(2),  # ball_or_not
            Discrete(2),  # last_possession
            Box(low=self.field_bounds_x, high=self.field_bounds_x+self.field_width, shape=(2,)),  # ball_position
            Box(low=0, high=20, shape=(2,))  # ball_velo
        ))


    # Reset variables and enviornment
    def reset(self):
        # Reset all player positions, ball position, etc...
        player_positions = [0 for i in range(self.num_players * 2)]
        half_num_players = self.num_players // 2
        if verbose:
            print("player pos - reset func:", player_positions)
        # Reset player positions
        for i in range(half_num_players):
            player_positions[i * 2] = self.field_bounds_x + (self.field_width // 4)
            player_positions[(i * 2) + 1] = self.field_bounds_y + (self.field_height // 2) + (i * 20)
        for i in range(half_num_players):
            player_positions[(i + half_num_players) * 2] = self.field_bounds_x + (self.field_width // 4) * 3
            player_positions[((i + half_num_players) * 2) + 1] = self.field_bounds_y + (self.field_height // 2) + (i * 20)
        player_velos = [[0, 0] for i in range(self.num_players)]


        player_orientations = [0 for i in range(self.num_players)] # Reset orientation
        player_sprint_bars = [self.total_sprint_bar for i in range(self.num_players)] # Reset sprint bars
        ball_or_not = [0 for i in range(self.num_players)] # Reset ball or not
        last_possession = [0, 0] # Reset last possession
        ball_position = [self.field_bounds_x + self.field_width // 2, self.field_bounds_y + self.field_height // 2] # Reset ball position
        ball_velo = [0, 0]

        return_obs = self.return_obs([player_positions, player_orientations, player_velos, player_sprint_bars, ball_or_not, last_possession, ball_position, ball_velo])
        return return_obs, len(return_obs)



    # Player orientation will be in radians. 0 radians will be facing to the "right" in the enviornment
    def step(self, obs, player_actions, timestep, render_mode):
        # Split obs
        player_positions, player_orientations, player_velos, ball_or_not, player_sprint_bars, last_possession, ball_position, ball_velo = self.split_obs(obs)
        if verbose:
            print("183 check player positions:", player_positions)
            print("184 last posession:", last_possession)
            print("185 ball or not", ball_or_not)
        # reset step_reward
        self.step_reward = [0, 0]

        if timestep == 0 and render_mode == True:
            print("Render mode: Human")
            pygame.init()
            pygame.display.set_caption('Soccer')
            self.display = pygame.display.set_mode((screen_width_ft, screen_height_ft))
            self.font = pygame.font.SysFont('Arial_bold', 380)

            self.Players = []
            player = Player()
            # Create a Player object
            for i in range(self.num_players):
                self.Players.append(copy.deepcopy(player))
            # Create ball
            self.ball = Ball()

        # Check whether ball is in goal or out of bounds or in goal
        terminated, last_possession, player_positions, player_velos, ball_or_not = self.check_ball_position(player_positions, player_velos, ball_position, last_possession)
        if verbose:
            print("190 check player positions:", player_positions)
            print("191 last posession:", last_possession)
            print("192 ball or not", ball_or_not)


        # Find reward based off of location
        reward = self.ball_location_reward(ball_position)
        self.step_reward = [self.step_reward[i] + reward[i] for i in range(len(self.step_reward))]

        # Give negative rewards if agent it out of bounds
        self.out_of_bounds_reward(player_positions)

        # Find new player and ball positions
        player_positions, player_velos, player_orientations, player_sprint_bar = self.player_position(player_positions, player_orientations, player_velos, player_actions, player_sprint_bars)
        ball_position, ball_velo = self.move_ball(player_positions, player_orientations, ball_or_not, ball_position, ball_velo)
        if verbose:
            print("197 check player positions:", player_positions)
            print("203 ball or not", ball_or_not)


        # Check for collisions
        player_positions, player_velos = self.player_collision(player_positions, player_velos)
        if verbose:
            print("201 check player positions:", player_positions)

        # Pass ball
            print("player actions - pass ball:", player_actions)
            print("212 ball or not - pass ball:", ball_or_not)
        for j in range(2):
            for i in range(self.num_players // 2):
                if verbose:
                    print("i and j:", i, j)
                if ball_or_not[i] == 1 and player_actions[j][i][3] == 1:
                    ball_velo = self.pass_ball(player_orientations[i])
        # Check tackles
        ball_or_not = self.get_tackled(player_positions, player_orientations, player_actions[0], ball_or_not)
        ball_or_not = self.get_tackled(player_positions, player_orientations, player_actions[1], ball_or_not)

        # calculate new ball velo based on friction
        if ball_velo[0] > 0 or ball_velo[1] > 0:
            ball_velo = [(self.meu_friction_force) + ball_velo[i] for i in range(2)]
        if verbose:
            print("224 check player positions:", player_positions)
            print("224 ball or not", ball_or_not)

        # TODO: offsides?

        # Render
        if render_mode == True:
            self.render(player_positions, player_orientations, player_velos, player_sprint_bars, ball_position, ball_velo, timestep)
            time.sleep(.1)
        # Check max timesteps
        truncated = self.terminated_or_not(timestep)
        if verbose:
            print("step reward:", torch.tensor(self.step_reward))
        return self.return_obs([player_positions, player_orientations, player_velos, player_sprint_bars, ball_or_not, last_possession, ball_position, ball_velo]), torch.tensor(self.step_reward), torch.tensor(terminated), torch.tensor(truncated)

    # Convert discrete actions to binary list to pass into step function
    def convert_discrete_action_to_binary_list(self, action):
        # Make is so it's like [[positions], [orientations]... ]
        total_actions = []
        player_velos = []
        player_orientations = []
        player_sprinting = []
        player_tackling = []
        player_passing = []
        for i in range(self.num_players):
            for j in range(5): # Number of possible of actions
                player_velos.append(1 if action[i][j] == k else 0 for k in range(self.action_space.spaces[i][j].n))
                player_orientations.append(1 if action[i][j] == k else 0 for k in range(self.action_space.spaces[i][j].n))
                player_tackling.append(1 if action[i][j] == k else 0 for k in range(self.action_space.spaces[i][j].n))
                player_sprinting.append(1 if action[i][j] == k else 0 for k in range(self.action_space.spaces[i][j].n))
                player_passing.append(1 if action[i][j] == k else 0 for k in range(self.action_space.spaces[i][j].n))
        return [list(player_velos), list(player_orientations), list(player_sprinting), list(player_tackling), list(player_passing)]

    # Check whether the episode reached max length
    def terminated_or_not(self, timestep):
        if timestep >= self.max_steps:
            print("Max timesteps reached")
            return True
        else:
            return False

    # distance function
    def find_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Check whether ball is in goal or out of bounds
    # Last possession should be a 2 length list.
    def check_ball_position(self, player_positions, player_velos, ball_position, last_possession):
        if verbose:
            print("260 check player positions:", player_positions)
        new_ball_position = ball_position
        new_player_positions = player_positions
        new_player_velos = player_velos
        new_ball_or_not = [0 for i in range(self.num_players)]
        if verbose:
            print("264 check player positions:", new_player_positions)
            print("276 new ball or not:", new_ball_or_not)

        # If ball is in left goal
        if ball_position[0] < self.center_goal_position_x1 - 12 and ball_position[0] > self.center_goal_position_x1 + 12 and ball_position[1] < self.center_goal_position_y:
            terminated = True
            self.step_reward[0] -= 1
            self.step_reward[1] += 1

        # If ball is in right goal
        elif ball_position[0] < self.center_goal_position_x1 - 12 and ball_position[0] > self.center_goal_position_x1 + 12 and ball_position[1] < self.center_goal_position_y:
           terminated = True
           self.step_reward[0] += 1
           self.step_reward[1] -= 1

        else:
            terminated = False

        # If ball is out for a corner kick or goal kick
        if ball_position[0] < self.field_bounds_x or ball_position[0] > self.field_width + self.field_bounds_x:
            new_possession = last_possession[::-1] # create new posession

            # Define new x coordinate for ball depending on which side it went out on
            if ball_position[0] < self.field_bounds_x:
                if ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x, self.field_bounds_y]
                elif ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x, self.field_bounds_y + self.field_height]

            elif ball_position[0] > self.field_width + self.field_bounds_x:
                if ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x + self.field_width, self.field_bounds_y]
                elif ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x + self.field_width, self.field_bounds_y + self.field_height]

            # Rewards based on possession
            if new_possession == [1, 0]:
                self.step_reward[0] += .01
                self.step_reward[1] -= .01
            else:
                self.step_reward[0] += .01
                self.step_reward[1] -= .01

            new_player_positions, new_player_velos, new_ball_or_not = self.corner_kick_or_goal_kick(new_possession, player_positions, player_velos, new_ball_position)
            if verbose:
                print("306 check player positions:", new_player_positions)
                print("319 new ball or not:", new_ball_or_not)


        # If ball is out for a throwin
        elif ball_position[1] < self.field_bounds_y or ball_position[1] > self.field_height + self.field_bounds_y:
            new_possession = last_possession[::-1]
            # Create new ball position so ball isn't out of bounds
            if ball_position[1] < self.field_bounds_y: 
                new_ball_position[1] = self.field_bounds_y
            else:
                new_ball_position[1] = self.field_bounds_y + self.field_height
            new_player_positions, new_player_velos, new_ball_or_not = self.throwin(new_possession, player_positions, player_velos, new_ball_position)
            if verbose:
                print("317 check player positions:", new_player_positions)
                print("330 new ball or not:", new_ball_or_not)


            # Rewards based on possession
            if new_possession == [1, 0]:
                self.step_reward[0] += .005
                self.step_reward[1] -= .005
            else:
                self.step_reward[0] += .005
                self.step_reward[1] -= .005
        else:
            new_possession = last_possession
        if verbose:
            print("329 check player positions:", new_player_positions)

        return terminated, new_possession, new_player_positions, new_player_velos, new_ball_or_not


    # Define throwin function
    # need to define which player should take it
    def throwin(self, new_possession, player_positions, player_velos, ball_position):
        # Find closest player
        closest_player = self.closest_player(new_possession, player_positions, ball_position)

        # Define new player positions and velos and who has the ball
        new_player_positions = player_positions
        new_player_positions[closest_player * 2] = ball_position[0]
        new_player_positions[(closest_player * 2) + 1] = ball_position[1]


        new_player_velos = player_velos
        new_player_velos[closest_player * 2] = 0
        new_player_velos[(closest_player * 2) + 1] = 0

        # Subtract one from ball_or_not list so i don't have to delete a 0
        new_ball_or_not = [0 for i in range(self.num_players - 1)]
        new_ball_or_not.insert(closest_player, 1)

        return new_player_positions, new_player_velos, new_ball_or_not

    # Define kickin function
    # define which player should take it (closest)
    def corner_kick_or_goal_kick(self, new_possession, player_positions, player_velos, new_ball_position):
        # define new y position
        if new_possession == [1, 0]:
            # Check if it's on left or right side of field
            if new_ball_position[0] < self.field_width:
                new_ball_position[1] = 6

            # Check if it's bottom or top
            elif new_ball_position[1] > self.field_height:
                new_ball_position[1] = self.field_bounds_y
            else:
                new_ball_position[1] = self.field_bounds_y + self.field_height

        # find closest player and give posession to them
        closest_player = self.closest_player(new_possession, player_positions, new_ball_position)

        # Define new player positions and velos and who has the ball
        new_player_positions = player_positions
        new_player_positions[closest_player * 2] = new_ball_position[0]
        new_player_positions[(closest_player * 2) + 1] = new_ball_position[1]

        new_player_velos = player_velos
        new_player_velos[closest_player * 2] = 0
        new_player_velos[(closest_player * 2) + 1] = 0


        # Subtract one from ball_or_not list so i don't have to delete a 0
        new_ball_or_not = [0 for i in range(self.num_players - 1)]
        new_ball_or_not.insert(closest_player, 1)

        return new_player_positions, new_player_velos, new_ball_or_not

    # find closest player to the ball depending on who has possession
    def closest_player(self, new_possession, player_positions, ball_position):
        other_team = self.num_players // 2
        shortest_distance = 100_000_000_000 # placeholder value
        if new_possession == [1, 0]:
            for i in range(self.num_players // 2):
                if verbose:
                    print("1 find distance info - closest player func:", player_positions[(i + other_team) * 2],
                        player_positions[((i + other_team) * 2) + 1],
                        ball_position[0], ball_position[1])
                distance = self.find_distance(player_positions[i * 2], player_positions[(i * 2) + 1], ball_position[0],
                                              ball_position[1])

                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_player = i
        else:
            for i in range(self.num_players // 2):
                if verbose:
                    print("2 find distance info - closest player func:", player_positions, ball_position, "i + other_team * 2:", (i + other_team) * 2, "other_team:", other_team, "i:", i)
                distance = self.find_distance(player_positions[(i + other_team) * 2], player_positions[((i + other_team) * 2) + 1], ball_position[0], ball_position[1])

                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_player = i + other_team
        try:
            return closest_player
        except:
            # print("closest_player:", closest_player)
            print("player positions:", player_positions)
            print("ball position:", ball_position)
            print("Distance:", distance)
            print("shortest distance:", shortest_distance)
            raise UnboundLocalError

    # Calculate closness to goals and return rewards based on that
    def ball_location_reward(self, ball_position):
        if ball_position[0] > (self.field_width/2) + self.field_bounds_x:
            # distance formula stuff and fraction of that for reward
            distance = self.find_distance(ball_position[0], ball_position[1], self.center_goal_position_x2, self.center_goal_position_y)
            if 1 / (2 * distance) < .1:
                return [1 / (2 * distance), -1 / (2 * distance)]
            else:
                return [0.1, -0.1]
        else:
            distance = self.find_distance(ball_position[0], ball_position[1], self.center_goal_position_x1, self.center_goal_position_y)
            if 1 / (2 * distance) < .1:
                return [-1 / (2 * distance), 1 / (2 * distance)]
            else:
                return [-0.1, 0.1]

    # Calculate reward for going out of bounds
    def out_of_bounds_reward(self, player_positions):
        for i in range(self.num_players // 2):
            player_position0 = [player_positions[i * 2], player_positions[(i * 2) + 1]]
        for i in range(self.num_players // 2, self.num_players):
            player_position1 = [player_positions[i * 2], player_positions[(i * 2) + 1]]

        # define bounds for negative rewards
        if player_position0[0] < self.field_bounds_x or player_position0[0] > self.field_width + self.field_bounds_x:
            self.step_reward[0] -= 0.05

        if player_position0[1] < self.field_bounds_y or player_position0[1] > self.field_height + self.field_bounds_y:
            self.step_reward[0] -= 0.05

        if player_position1[0] < self.field_bounds_x or player_position1[0] > self.field_width + self.field_bounds_x:
            self.step_reward[1] -= 0.05

        if player_position1[1] < self.field_bounds_y or player_position1[1] > self.field_height + self.field_bounds_y:
            self.step_reward[1] -= 0.05


    # Pass ball function: calculate new velo for ball
    def pass_ball(self, player_orientation):

        new_x_velo = np.cos(player_orientation) * self.pass_velo
        new_y_velo = np.sin(player_orientation) * self.pass_velo

        return [new_x_velo, new_y_velo]

    # Calculate to see whether the player orientations line up within 45 degrees of eachother (90 degrees total)
    # Check relative closness
    def check_orientation(self, player1_orientation, player1_position, player2_orientation, player2_position):
        distance = self.find_distance(player1_position[0], player1_position[1], player2_position[0], player2_position[1])
        if player1_orientation + (np.pi / 2) > player2_orientation and player1_orientation - (np.pi / 2) < player2_orientation:
            return distance, True
        else:
            return distance, False

    # Get the ball if ball is infront and enemy player doesn't have the ball
    # Within 45 degrees (90 deg total) and distance < 5
    def get_ball(self, player_positions, player_position, player_orientation, ball_position):
        distance, facing = self.check_orientation(player_orientation, player_position, ball_position, np.atan(ball_position[1]/ball_position[0]))
        if distance < 5 and facing:
            # Subtract one from ball_or_not list so i don't have to delete a 0
            new_ball_or_not = [0 for i in range(self.num_players - 1)]
            try:
                new_ball_or_not.insert(player_positions.index(player_position), 1) # find index of player inside of player_positions

                # Check which team has possession
                if player_positions.index(player_position) > len(player_positions) // 2:
                    last_possession = [0, 1]
                else:
                    last_possession = [1, 0]
            except:
                new_ball_or_not = [0 for i in range(self.num_players)] # This should not be reached
                print("------------------------------")
                print("--- ERROR IN get_ball func ---")
                print("------------------------------")

            return new_ball_or_not, last_possession

    # Get tackled if enemy is close and tackles and player has ball
    # player_tackled_or_not should be a whole list of 0s and 1s determining if they decided to or not
    def get_tackled(self, player_positions, player_orientation, player_actions, new_possession):
        # Initialize an empty list to store the new possession
        if verbose:
            print("player actions - get tackled func:", player_actions)

        # Loop through each player
        for i in range(self.num_players // 2):
            if verbose:
                print(f"{i} player action:", player_actions[i])
            # Check if the player wants to tackle
            if player_actions[i][3] == 1:
                # Get the player's orientation and position
                player1_orientation = player_orientation[i]
                player1_position = [player_positions[(i * 2)], player_positions[(i * 2) + 1]]

                # Loop through the other players
                for j in range(self.num_players):
                    # Check if the other player has the ball
                    if new_possession[j]:
                        # Get the other player's orientation and position
                        player2_orientation = player_orientation[j]
                        player2_position = [player_positions[(j * 2)], player_positions[(j * 2) + 1]]

                        # Check if the two players are facing each other and are within 5
                        distance, facing_each_other = self.check_orientation(player1_orientation, player1_position, player2_orientation, player2_position)
                        if distance <= 5 and facing_each_other:
                            # The player has tackled the ball
                            new_possession[i] = 1
                            new_possession[j] = 0

                            # Give rewards to player who tackled successfully
                            if player_positions.index(player1_position) >= len(player_positions) // 2:
                                self.step_reward[0] -= .05
                                self.step_reward[1] += .05
                                # print("reward from tackling")
                            else:
                                self.step_reward[0] += .05
                                self.step_reward[1] -= .05
                                # print("reward from tackling")

                        else:
                            # If player is on second team
                            if player_positions.index(player1_position[0]) >= len(player_positions) // 2:
                                self.step_reward[0] += .05
                                self.step_reward[1] -= .05
                                # print("reward from tackling")


                            else:
                                self.step_reward[0] -= .05
                                self.step_reward[1] += .05
                                # print("reward from tackling")


        return new_possession

    # Find what player action is taken and add to player velo
    # assuming player_action is a list len 6 binary and 1 and 0th index
    # Find new player position depending on position, velo, and action
    # Cap out the velo to a certain amount, unless using sprint
    # Then find new player orientation
    def player_position(self, player_positions, player_orientations, player_velos, player_actions, player_sprint_bars):
        new_orientations = []
        player_action_velos = []
        new_player_sprint_bars = []
        player_actions = [action for sublist in player_actions for action in sublist]
        if verbose:
            print("player actions:", player_actions, "player velos:", player_velos)
        for i in range(self.num_players):
            if verbose:
                print("i:", i, f"player {i}th action:", player_actions[i])
                print("player velos given:", [player_velos[i * 2], player_velos[(i * 2) + 1]])
            # Find new player velo and orientation
            player_action_velo, new_orientation = self.generate_velocity_vector_and_orientation(player_actions[i], [player_velos[i * 2], player_velos[(i * 2) + 1]], player_orientations[i])

            # Check if player is sprinting
            if player_actions[i][2]:
                new_player_sprint_bar = player_sprint_bars[i] - 5
            else:
                new_player_sprint_bar = player_sprint_bars[i] + 2

            # Add new orientations, velos, and sprint bar to list
            new_orientations.append(new_orientation)
            player_action_velos.append(player_action_velo)
            new_player_sprint_bars.append(new_player_sprint_bar)

        # Calculate new pos based on new velo
        if verbose:
            print("player positions:", player_positions, "player velos:", player_velos)
        new_pos = [player_positions[i] + player_velos[i] for i in range(len(player_positions))]

        return new_pos, player_action_velos, new_orientations, new_player_sprint_bars

    def generate_velocity_vector_and_orientation(self, player_action, player_velo, player_orientation):
        # Define the angles in radians
        angles_in_radians = [np.pi / 12, 0, -np.pi / 12]
        pos_angles_in_radians = [np.pi / 12, 0, -np.pi / 12, np.pi]

        # Find the angle based on the input index
        angle = pos_angles_in_radians[player_action[0]]

        if verbose:
            print("calculate velos:", [np.cos(angle) * self.player_speed, np.sin(angle) * self.sprint_speed])
        # Check if player is sprinting or not
        if player_action[2]:
            # Create the velocity vector and multiply by player sprint speed
            velocity = np.array([np.cos(angle) * self.player_speed, np.sin(angle) * self.sprint_speed])
        else:
            # Create the velocity vector and multiply by player speed
            velocity = np.array([np.cos(angle) * self.player_speed, np.sin(angle) * self.player_speed])

        # Find angle based on input index
        angle = angles_in_radians[player_action[1]]

        # Find new orientation
        new_orientation = player_orientation + angle
        while True:
            if new_orientation > np.pi * 2:
                new_orientation = new_orientation - 2 * np.pi
            elif new_orientation < 0:
                new_orientation = new_orientation + 2 * np.pi
            else:
                break

        if verbose:
            print("velocity:", velocity, "player_velo,", player_velo)
        final_velo = [velocity[0] + player_velo[0], velocity[1] + player_velo[1]]
        mag_final_velo = self.find_distance(velocity[0], velocity[1], 0, 0)
        if mag_final_velo > self.player_max_speed:
            # Normalize and then multiply by max_speed
            final_velo[0] = (final_velo[0] // mag_final_velo) * self.player_max_speed
            final_velo[1] = (final_velo[1] // mag_final_velo) * self.player_max_speed


        return final_velo, new_orientation

    # Move ball depending on players and ball velo if no one has it
    # Player positions should be new positions so we can move ball to player
    def move_ball(self, player_positions, player_orientations, ball_or_not, ball_position, ball_velo):
        new_ball_position = ball_position
        # Check if someone has possession
        if any(element == 1 for element in ball_or_not):
            # Find index of possession
            for i in range(len(ball_or_not)):
                if ball_or_not[i]:
                    index = i
            if verbose:
                print("index:", index)
                print("player_orientations:", player_orientations)
            # Calculate ball position based on player position and orientation of player
            new_ball_position[0] = player_positions[(index * 2)] + 3 * np.cos(player_orientations[index])
            new_ball_position[1] = player_positions[(index * 2) + 1] + 3 * np.cos(player_orientations[(index)])

            new_ball_velo = [0, 0]
            return new_ball_position, new_ball_velo
        else:
            new_ball_position = [ball_position[i] + ball_velo[i] for i in range(len(ball_position))]
            return new_ball_position, ball_velo

    # Check to see if players are in same space
    # Move one half way out back in direction of velo
    # use hal of that space because when looping again it will move the other one half way out.
        # if they do set velos to very low
    def player_collision(self, player_positions, player_velos):
        new_player_positions = player_positions
        new_player_velos = player_velos
        other_team = self.num_players // 2
        if verbose:
            print("player positions:", player_positions)
            print("player velos:", player_velos)
        for i in range(self.num_players // 2):
            for j in range(self.num_players // 2):
                # Find distance between players
                distance = self.find_distance(player_positions[i * 2], player_positions[(i * 2) + 1], player_positions[(j + other_team) * 2], player_positions[((j + other_team) * 2) + 1])
                if distance < 3:
                    # find magnitude for vectors through distance formulas
                    normalize = self.find_distance(player_velos[i * 2], player_velos[(i * 2) + 1], 0, 0)

                    # Calculate new player positions and velos
                    new_player_positions[i * 2] = [player_positions[i * 2] + (player_velos[i * 2]/normalize)]
                    new_player_positions[(i * 2) + 1] = [player_positions[(i * 2) + 1] + (player_velos[(i * 2) + 1]/normalize)]

                    new_player_positions[j + other_team * 2] = [player_positions[(j + other_team) * 2] + (player_velos[(j + other_team) * 2]/normalize)]
                    new_player_positions[(j + other_team * 2) + 1] = [player_positions[((j + other_team) * 2) + 1] + (player_velos[((j + other_team) * 2) + 1]/normalize)]

                    new_player_velos[i * 2] = 0
                    new_player_velos[(i * 2) + 1] = 0
                    new_player_velos[((j + other_team) * 2)] = 0
                    new_player_velos[((j + other_team) * 2) + 1] = 0

        return new_player_positions, new_player_velos

    # Offsides function
    def offsides(self, player_positions, ball_or_not):
        # Get the player who has the ball
        ball_holder_index = ball_or_not.index(1)
        # ball_holder_position = player_positions[ball_holder_index]

        # Get the positions of all of the players on the opposing team and find the last 2 defenders
        other_team = self.num_players // 2
        if ball_holder_index > other_team:
            opposing_team_positions = [player_positions[i] for i in range(other_team)]

            # Find last 2 defenders
            last_defender = player_positions[other_team * 2]
            second_last_defender = player_positions[(other_team * 2) + 1]
            for i in range(other_team - 2):
                if player_positions[(i + 2 + other_team) * 2] < second_last_defender:
                    if player_positions[(i + 2 + other_team) * 2] < last_defender:
                        second_last_defender = last_defender
                        last_defender = player_positions[(i + 2 + other_team) * 2]
                    else:
                        second_last_defender = player_positions[(i + 2 + other_team) * 2]

        else:
            opposing_team_positions = [player_positions[i + other_team] for i in range(other_team)]

            # Find last 2 defenders
            last_defender = player_positions[0]
            second_last_defender = player_positions[1]
            for i in range(other_team - 2):
                if player_positions[(i + 2) * 2] < second_last_defender:
                    if player_positions[(i + 2) * 2] < last_defender:
                        second_last_defender = last_defender
                        last_defender = player_positions[(i + 2) * 2]
                    else:
                        second_last_defender = player_positions[(i + 2) * 2]

        # Check if any of the players on the opposing team are offside
        offside_players = []
        for opposing_team_player_position in opposing_team_positions:
            for i in range(other_team):
                # Check if the opposing team player is closer to the ball than the second-to-last defender (i.e. the goalkeeper)
                if opposing_team_player_position[0] < second_last_defender:
                    self.step_reward[0] -= .1
                    self.step_reward[1] += .1

    # Split obs
    def split_obs(self, obs):
        obs0 = []
        obs1 = []
        obs2 = []
        obs3 = []
        obs4 = []
        obs5 = []
        obs6 = []
        obs7 = []
        if verbose:
            print("obs: ", obs)
        # player_positions, player_orientations, player_velos, player_sprint_bars, ball_or_not, last_possession, ball_position, ball_velo
        # print("Observations:", obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7])
        for i in range(len(obs)):
            # Check if it's player positions
            if i < 2 * self.num_players:
                obs0.append(obs[i])
                # print("AHH 0", obs[i])

            # Check for player orientations
            elif i < 3 * self.num_players:
                obs1.append(obs[i])
                # print("AHH 1", obs[i])

            # Check for player velos
            elif i < 5 * self.num_players:
                obs2.append(obs[i])
                # print("AHH 2", obs[i])

            # Check for player sprint bars
            elif i < 6 * self.num_players:
                obs3.append(obs[i])
                # print("AHH 3", obs[i])

            # Check for player possession
            elif i < 7 * self.num_players:
                obs4.append(obs[i])
                # print("AHHJ 4", obs[i])

            # Check for last possession
            elif i < 7 * self.num_players + 2:
                obs5.append(obs[i])
                # print("AHH 5", obs[i])

            # Check for ball position
            elif i < 7 * self.num_players + 4:
                obs6.append(obs[i])
                # print("AHH 6", obs[i])

            # Check for ball velo
            elif i < 7 * self.num_players + 6:
                obs7.append(obs[i])
                # print("AHH 7", obs[i])

        split_obs_new = [obs0, obs1, obs2, obs3, obs4, obs5, obs6, obs7]
        if verbose:
            print("splitting obs: ")
            print("obs 0 (player pos):", split_obs_new[0])
            print("obs 1 (player orientation):", split_obs_new[1])
            print("obs 2 (player velo):", split_obs_new[2])
            print("obs 3 (sprint bars):", split_obs_new[3])
            print("obs 4 (ball or not):", split_obs_new[4])
            print("obs 5 (last possession):", split_obs_new[5])
            print("obs 6 (ball pos):", split_obs_new[6])
            print("obs 7 (ball velo):", split_obs_new[7])
            print("returning split obs:", split_obs_new)

        return split_obs_new

    # Return obs to network as a 1d list
    def return_obs(self, obs):
        if verbose:
            print("pre-returning obs:", obs)
        new_list = [(self.flatten_list_of_lists(obs[0]))]
        new_list.append(obs[1])
        new_list.append(self.flatten_list_of_lists(obs[2]))
        new_list.append(obs[3])
        new_list.append(obs[4])
        new_list.append(obs[5])
        new_list.append(obs[6])
        new_list.append(obs[7])

        if verbose:
            print("returning obs")
            print("obs 0 (player pos):", obs[0])
            print("obs 1 (player orientation):", obs[1])
            print("obs 2 (player velo):", obs[2])
            print("obs 3 (sprint bars):", obs[3])
            print("obs 4 (ball or not):", obs[4])
            print("obs 5 (last possession):", obs[5])
            print("obs 6 (ball pos):", obs[6])
            print("obs 7 (ball velo):", obs[7])

        new_list = torch.tensor(self.flatten_list_of_lists(new_list))
        if verbose:
            print("returning obs:", new_list)

        return new_list

    # Flatten 2d lists into 1d lists
    def flatten_list_of_lists(self, nested_list):
        flat_list = []
        for sublist in nested_list:
            try:
                for item in sublist:
                    try:
                        flat_list.append(item.tolist())
                    except AttributeError:
                        flat_list.append(item)
                    except:
                        print("Error! in Flatten List of Lists")
            except:
                return nested_list
        return flat_list

    def render(self, player_positions, player_orientations, player_velos, player_sprint_bars, ball_position, ball_velo, timestep, mode='human'):
        screen = pygame.display.set_mode([screen_width_ft, screen_height_ft])
        # Fill the background with green
        screen.fill((50, 168, 82))

        # Draw field
        # Draw outside box
        pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_width, box_height), 3)  # Black bounding line, 3 pixels thick
        pygame.draw.rect(screen, (50, 168, 82), (box_x + 3, box_y + 3, box_width - 6, box_height - 6))  # green
        # Center circle
        pygame.draw.circle(screen, (0, 0, 0), (screen_width_ft // 2, screen_height_ft // 2), 30)
        pygame.draw.circle(screen, (50, 168, 82), (screen_width_ft // 2, screen_height_ft // 2), 28)
        # Draw midline
        pygame.draw.rect(screen, (0, 0, 0), (center_x - 1, center_y - (box_height / 2), 2, box_height))
        # Draw goals
        # Goal #1:
        pygame.draw.rect(screen, (0, 0, 0), (box_x - 17, center_y - 24, 17, 48))
        # Goal #2:
        pygame.draw.rect(screen, (0, 0, 0), (box_x + box_width, center_y - 24, 17, 48))
        # Move and draw the player
        if verbose:
            print("player velos:", player_velos)
        if timestep == 0:
            return

        for i in range(self.num_players):
            if i < self.num_players // 2:
                team = 0
            else:
                team = 1
            self.Players[i].move(player_positions[i * 2], player_positions[(i * 2) + 1], [player_velos[i]], player_orientations[i], team)


        for i in range(self.num_players):
            self.Players[i].draw(screen)

        self.ball.move(ball_position[0], ball_position[1], ball_velo)
        self.ball.draw(screen)

        # Flip the display
        pygame.display.flip()

    def close(self):
        pygame.quit()
