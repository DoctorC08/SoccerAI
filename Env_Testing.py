# Env_Testing.py
# by christophermao
# 12/22/23
import math

# import libraries
import numpy as np

import pygame
import gym
from gym import spaces
from gym.spaces import Box, Tuple, Discrete

import numpy as np
import torch

import copy

import time

global verbose
verbose = False

global reward_verbose
reward_verbose = True

# Pygame stuff for rendering
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
    def __init__(self, starting_x_pos, starting_y_pos):
        self.radius = 8  # Radius for the circle
        self.x = center_x
        self.y = center_y
        self.x_velo = starting_x_pos
        self.y_velo = starting_y_pos
        self.team = 0

    def move(self, x_pos, y_pos, velo, team):
        if verbose:
            print("velo:", velo)
            print(x_pos, self.x, y_pos, self.y)
        self.x = self.x + float(x_pos)
        self.y = self.y + float(y_pos)
        # self.x_velo = float(velo[0])
        # self.y_velo = float(velo[1])
        self.team = float(team)
        return self.x, self.y

    def draw(self, surface):
        # Draw player position
        if self.team == 0:
            pygame.draw.circle(surface, (0, 0, 255), (self.x, self.y), self.radius)
        else:
            pygame.draw.circle(surface, (255, 0, 0), (self.x, self.y), self.radius)

        # Draw player velo and orientation
        # pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + self.x_velo, self.y + self.y_velo), width=2) # Velo

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
            print("x and y", self.x, self.y)

        # Draw ball
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), self.radius)
        # Draw velocity vector
        # pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + self.x_velo, self.y + self.y_velo), width=2)

# Env

def closest_player(new_possession):
    other_team = num_players // 2
    shortest_distance = 100_000_000_000 # placeholder value
    if new_possession == [1, 0]:
        closest_player = 0
    else:
        closest_player = 1

    return closest_player

def split_obs(obs):
    obs0 = []
    obs1 = []
    obs2 = []
    obs3 = []
    obs4 = []
    obs5 = []
    if verbose:
        print("obs: ", obs)
        print("Observations:", obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7])
    for i in range(len(obs)):
        # Check if it's player positions
        if i < 2 * num_players:
            obs0.append(obs[i])
            # print("AHH 0", obs[i])

        # Check for player velos
        elif i < 4 * num_players:
            obs1.append(obs[i])
            # print("AHH 2", obs[i])

        # Check for player possession
        elif i < 5 * num_players:
            obs2.append(obs[i])
            # print("AHHJ 4", obs[i])

        # Check for last possession
        elif i < 5 * num_players + 2:
            obs3.append(obs[i])
            # print("AHH 5", obs[i])

        # Check for ball position
        elif i < 5 * num_players + 4:
            obs4.append(obs[i])
            # print("AHH 6", obs[i])

        # Check for ball velo
        elif i < 5 * num_players + 6:
            obs5.append(obs[i])
            # print("AHH 7", obs[i])

    split_obs_new = [obs0, obs1, obs2, obs3, obs4, obs5]
    if verbose:
        print("splitting obs: ")
        print("obs 0 (player pos):", split_obs_new[0])
        print("obs 2 (player velo):", split_obs_new[1])
        print("obs 3 (ball or not):", split_obs_new[2])
        print("obs 4 (last possession):", split_obs_new[3])
        print("obs 5 (ball pos):", split_obs_new[4])
        print("obs 6 (ball velo):", split_obs_new[5])
        print("returning split obs:", split_obs_new)

    return split_obs_new

def flatten_list_of_lists(nested_list):
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
                    raise EnvironmentError
        except:
            return nested_list
    return flat_list

def close():
    pygame.quit()

timestep = 0
state = 0  # Initial state

# Define field widths and heights and bounds
field_width = 345 * 2
field_height = 225 * 2
field_bounds_x = 10 * 2
field_bounds_y = 20 * 2

# multiplied by 2 for scale divided for half
center_y = (field_height + field_bounds_y * 2) // 2

# define goal positions
center_goal_position_x1 = field_bounds_x
center_goal_position_x2 = field_bounds_x + field_width

# y coordinate is same for both
center_goal_position_y = (field_height / 2) + (field_bounds_y)

# Define number of players (both teams)
num_players = 2
# Define player speed
player_speed = 5
# Define max player speed
player_max_speed = 20

# Step function!
step_reward = [0, 0]

#define timesteps
max_steps = 100
current_step = 0

total_ball_loc_rewardsa = 0
total_ball_loc_rewardsb = 0


def return_obs(obs):
    if verbose:
        print("pre-returning obs:", obs)
    new_list = [(flatten_list_of_lists(obs[0]))]
    new_list.append(flatten_list_of_lists(obs[1]))
    new_list.append(obs[2])
    new_list.append(obs[3])
    new_list.append(obs[4])
    new_list.append(obs[5])

    if verbose:
        print("returning obs")
        print("obs 0 (player pos):", obs[0])
        print("obs 2 (player velo):", obs[1])
        print("obs 4 (ball or not):", obs[2])
        print("obs 5 (last possession):", obs[3])
        print("obs 6 (ball pos):", obs[4])
        print("obs 7 (ball velo):", obs[5])

    new_list = torch.tensor(flatten_list_of_lists(new_list))
    if verbose:
        print("returning obs:", new_list)

    return new_list

def reset():
    # Reset all player positions, ball position, etc...
    player_positions = [0 for i in range(num_players * 2)]
    half_num_players = num_players // 2
    if verbose:
        print("player pos - reset func:", player_positions)
    # Reset player positions
    for i in range(half_num_players):
        player_positions[i * 2] = field_bounds_x + (field_width // 4)
        player_positions[(i * 2) + 1] = field_bounds_y + (field_height // 2) + (i * 20)
    for i in range(half_num_players):
        player_positions[(i + half_num_players) * 2] = field_bounds_x + (field_width // 4) * 3
        player_positions[((i + half_num_players) * 2) + 1] = field_bounds_y + (field_height // 2) + (i * 20)
    player_velos = [0 for i in range(num_players * 2)]

    ball_or_not = [0 for i in range(num_players)] # Reset ball or not
    last_possession = [0, 0] # Reset last possession
    ball_position = [field_bounds_x + field_width // 2, field_bounds_y + field_height // 2] # Reset ball position
    ball_velo = [0, 0]

    new_obs = return_obs([player_positions, player_velos, ball_or_not, last_possession, ball_position, ball_velo])
    return new_obs, len(new_obs)

def check_ball_position(player_positions, ball_position, last_possession):
    # Check if ball is in goal or not or out of bounds
    if verbose:
        print("260 check player positions:", player_positions)

        print("ball position:", ball_position)
    # If ball is in left goal
    if ball_position[0] < center_goal_position_x1 + 10 and ball_position[1] < center_goal_position_y + 30 and ball_position[1] > center_goal_position_y - 30:
        terminated = True
        step_reward[0] -= 2
        step_reward[1] += 2
        print("An Agent Scored! adding reward to team 2")

    # If ball is in right goal
    elif ball_position[0] > center_goal_position_x2 - 10 and ball_position[1] < center_goal_position_y + 30 and ball_position[1] > center_goal_position_y - 30:
       terminated = True
       step_reward[0] += 2
       step_reward[1] -= 2
       print("An Agent Scored! adding reward to team 1")

    # If ball is out of goal line
    elif ball_position[0] < field_bounds_x or ball_position[0] > field_bounds_x + field_width:
        terminated = True
        # Negative reward if you dribbled out:
        neg_reward_on_possession(last_possession, reward=0.5)
        last_possession.reverse()
        if reward_verbose:
            print("An Agent dribbled out of bounds")


    elif ball_position[1] < field_bounds_y or ball_position[1] > field_bounds_y + field_height:
        terminated = True
        # Neg reward if dribbled out
        neg_reward_on_possession(last_possession, reward=0.3)
        last_possession.reverse()
        if reward_verbose:
            print("An Agent dribbled out of bounds")

    else:
        terminated = False


    return terminated, last_possession

def neg_reward_on_possession(last_possession, reward):
    if last_possession == [0, 1]:
        step_reward[1] -= reward
        if reward_verbose:
            print("giving neg reward to team:", last_possession)
            print("neg reward out of bounds:", reward)
    else:
        step_reward[0] -= reward
        if reward_verbose:
            print("giving neg reward to team:", last_possession)
            print("neg reward out of bounds:", reward)

def get_ball(player_positions, ball_position, last_possession, ball_or_not):
    for i in ball_or_not:
        if i == 1:
            return ball_or_not, last_possession
        else:
            pass
    if verbose:
        print("player positions:", player_positions)
    for i in range(num_players):
        if verbose:
            print("\nplayer position:", player_positions[i * 2], player_positions[(i * 2) + 1])
            print("ball position:", ball_position[0], ball_position[1])
            print("distance:", find_distance(player_positions[i * 2], player_positions[(i * 2) + 1], ball_position[0], ball_position[1]))
        distance = find_distance(player_positions[i * 2], player_positions[(i * 2) + 1], ball_position[0], ball_position[1])
        if distance <= 50:
            if verbose:
                print("True")
            new_ball_or_not = [0 for i in range(num_players)]
            new_ball_or_not[i] = 1 # find index of player inside of player_positions
            if last_possession == [0, 0]:
                if i <= num_players / 2:
                    last_possession = [0, 1]
                else:
                    last_possession = [1, 0]
            else:
                last_possession.reverse()

            return new_ball_or_not, last_possession
    return ball_or_not, last_possession

def ball_location_reward(ball_position):
    if ball_position[0] > (field_width/2) + field_bounds_x:
        # distance formula stuff and fraction of that for reward
        distance = find_distance(ball_position[0], ball_position[1], center_goal_position_x2, center_goal_position_y)
        if reward_verbose:
            print("ball loc reward (anti-cut):", 1 / distance)
        if 1 / distance < 0.002:
            return [0, 0]
        elif 1 / (2 * distance) < .1:
            # print(" or here", 1 / (2 * distance))
            return [1 / distance, -1 / distance]
        else:
            return [0.1, -0.1]
    else:
        distance = find_distance(ball_position[0], ball_position[1], center_goal_position_x1, center_goal_position_y)
        if reward_verbose:
            print("ball loc reward (anti-cut):", 1 / distance)
        if 1 / distance < 0.002:
            return [0, 0]
        elif 1 / distance < .1:
            return [-1 / distance, 1 / distance]
        else:
            return [-0.1, 0.1]

def out_of_bounds_reward(player_positions):
    # Negative reward for the player going out of bounds
    for i in range(num_players // 2):
        player_position0 = [player_positions[i * 2], player_positions[(i * 2) + 1]]
    for i in range(num_players // 2, num_players):
        player_position1 = [player_positions[i * 2], player_positions[(i * 2) + 1]]

    # define bounds for negative rewards
    # DO NOT CHANGE TO ELIF
    if player_position0[0] < field_bounds_x or player_position0[0] > field_width + field_bounds_x:
        step_reward[0] -= 0.1
        if reward_verbose:
            print("Out of bounds reward to team 1")

    if player_position0[1] < field_bounds_y or player_position0[1] > field_height + field_bounds_y:
        step_reward[0] -= 0.1
        if reward_verbose:
            print("Out of bounds reward to team 1")


    if player_position1[0] < field_bounds_x or player_position1[0] > field_width + field_bounds_x:
        step_reward[1] -= 0.1
        if reward_verbose:
            print("Out of bounds reward to team 2")


    if player_position1[1] < field_bounds_y or player_position1[1] > field_height + field_bounds_y:
        step_reward[1] -= 0.1
        if reward_verbose:
            print("Out of bounds reward to team 2")

def move_ball(player_positions, ball_or_not, ball_position, ball_velo):
    new_ball_position = ball_position
    # Check if someone has possession
    if any(element == 1 for element in ball_or_not):
        # Find index of possession
        for i in range(len(ball_or_not)):
            if ball_or_not[i]:
                index = i
        if verbose:
            print("index:", index)
        # Calculate ball position based on player position and orientation of player
        new_ball_position[0] = player_positions[(index * 2)]
        new_ball_position[1] = player_positions[(index * 2) + 1]

        new_ball_velo = [0, 0]
        return new_ball_position, new_ball_velo
    else:
        new_ball_position = [ball_position[i] + ball_velo[i] for i in range(len(ball_position))]
        return new_ball_position, ball_velo

def find_distance(x1, y1, x2, y2):
    if verbose:
        print(x1, y1)
        print(x2, y2)
    try:
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except TypeError:
        try:
            return torch.sqrt((x1[0] - x2[0]) ** 2 + (y1[0] - y2[0]) ** 2)
        except:
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except:
        try:
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        except:
            print(x1, y1)
            print(x2, y2)
            raise EnvironmentError

def player_collision(player_positions, player_velos):
    new_player_positions = player_positions
    new_player_velos = player_velos
    other_team = num_players // 2
    if verbose:
        print("player positions:", player_positions)
        print("player velos:", player_velos)
    for i in range(num_players // 2):
        for j in range(num_players // 2):
            # Find distance between players
            distance = find_distance(player_positions[i * 2], player_positions[(i * 2) + 1], player_positions[(j + other_team) * 2], player_positions[((j + other_team) * 2) + 1])
            if distance < 20:
                if verbose:
                    # find magnitude for vectors through distance formulas
                    print("player velos", player_velos)
                normalize = find_distance(player_velos[i * 2], player_velos[(i * 2) + 1], 0, 0)

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

def get_tackled(player_positions, player_actions, new_possession):
    # print("player positions:", player_positions)
    # Initialize an empty list to store the new possession
    if verbose:
        print("possession:", new_possession)
        print("player actions - get tackled func:", player_actions)        # Loop through each player
    for i in range(num_players):
        if verbose:
            print(f"{i} player action:", player_actions[i])
        # Check if the player wants to tackle
        if player_actions[i] == 1 and new_possession[i] == 0:
            # Get the player's position
            player1_position = [player_positions[(i * 2)], player_positions[(i * 2) + 1]]

            if i < num_players // 2:
                for j in range(num_players // 2, num_players):
                    # Check if the other player has the ball
                    print(new_possession)
                    if new_possession[j]:
                        # Get the other player's orientation and position
                        player2_position = [player_positions[(j * 2)], player_positions[(j * 2) + 1]]
                        # print("player position1 and 2:", player1_position, player2_position)

                        # Check if the two players are facing each other and are within 5
                        distance = find_distance(player1_position[0], player1_position[1], player2_position[0],
                                                 player2_position[1])
                        if verbose:
                            print("distance:", distance)

                        if distance <= 30:
                            # The player has tackled the ball
                            new_possession[i] = 1
                            new_possession[j] = 0

                            # Give rewards to player who tackled successfully
                            step_reward[0] += .05
                            step_reward[1] -= .05
                            if reward_verbose:
                                print("reward from tacklinga player 1 adding reward to team 1")

                        else:
                            step_reward[0] -= .03
                            if reward_verbose:
                                print("reward from tacklingb player 1")
            else:
                # Loop through the other players
                for j in range(num_players // 2):
                    # Check if the other player has the ball
                    if new_possession[j]:

                        # Get the other player's orientation and position
                        player2_position = [player_positions[(j * 2)], player_positions[(j * 2) + 1]]

                        # Check if the two players are facing each other and are within 5
                        distance = find_distance(player1_position[0], player1_position[1], player2_position[0],
                                                 player2_position[1])
                        if verbose:
                            print("distance:", distance)

                        if distance <= 30:
                            # The player has tackled the ball
                            new_possession[i] = 1
                            new_possession[j] = 0

                            step_reward[0] -= .05
                            step_reward[1] += .05
                            if reward_verbose:
                                print("reward from tacklinga player 2 adding reward to team 2")

                        else:
                            step_reward[1] -= .03
                            if reward_verbose:
                                print("reward from tacklingb player 2")



    return new_possession

def render(player_positions, player_velos, ball_position, ball_velo, timestep, possession):
    player_positions = player_positions
    new_possession = possession
    # Fill the background with green
    screen.fill((50, 168, 82))

    # Draw field
    # Draw outside box
    pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_width, box_height),
                     3)  # Black bounding line, 3 pixels thick
    pygame.draw.rect(screen, (50, 168, 82), (box_x + 3, box_y + 3, box_width - 6, box_height - 6))  # green
    # Center circle
    pygame.draw.circle(screen, (0, 0, 0), (screen_width_ft // 2, screen_height_ft // 2), 30)
    pygame.draw.circle(screen, (50, 168, 82), (screen_width_ft // 2, screen_height_ft // 2), 28)
    # Draw midline
    pygame.draw.rect(screen, (0, 0, 0), (center_x - 1, center_y - (box_height / 2), 2, box_height))
    # Draw goals
    # Goal #1:
    pygame.draw.rect(screen, (0, 0, 0), (box_x - 17, center_y - 30, 17, 60))
    # Goal #2:
    pygame.draw.rect(screen, (0, 0, 0), (box_x + box_width, center_y - 30, 17, 60))

    # Move and draw the player
    if verbose:
        print("player velos:", player_velos)
    if timestep == 0:
        return player_positions, new_possession

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        keys = pygame.key.get_pressed()

        if event.type == pygame.KEYDOWN:
            for count, player in enumerate(Players):
                if count == 0:
                    # Blue player movement
                    if keys[pygame.K_LEFT]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(-50, 0, 0, 0)
                    if keys[pygame.K_RIGHT]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(50, 0, 0, 0)
                    if keys[pygame.K_UP]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(0, -50, 0, 0)
                    if keys[pygame.K_DOWN]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(0, 50, 0, 0)
                    if keys[pygame.K_l]:
                        if verbose:
                            print("Attempting to tackle - player positions:", player_positions)
                            print("Current possession:", possession)
                        new_possession = get_tackled(player_positions, [1, 0], possession)
                else:
                    # Red player movement
                    if keys[pygame.K_a]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(-50, 0, 0, 1)
                    if keys[pygame.K_d]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(50, 0, 0, 1)
                    if keys[pygame.K_w]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(0, -50, 0, 1)
                    if keys[pygame.K_s]:
                        player_positions[count * 2], player_positions[(count * 2) + 1] = player.move(0, 50, 0, 1)
                    if keys[pygame.K_SPACE]:
                        if verbose:
                            print("Attempting to tackle - player positions:", player_positions)
                            print("Current possession:", possession)
                        new_possession = get_tackled(player_positions, [0, 1], possession)

        # print("keys:", keys[pygame.K_RIGHT], keys[pygame.K_LEFT], keys[pygame.K_UP], keys[pygame.K_DOWN], "\n\n")
        # print("event:", event)
    # for i in range(num_players):
    #     if i < num_players // 2:
    #         team = 0
    #     else:
    #         team = 1
    #     Players[i].move(player_positions[i * 2], player_positions[(i * 2) + 1], [player_velos[i * 2], player_velos[(i * 2) + 1]], team)
    if verbose:
        print("players:", Players)
    for i in range(num_players):
        Players[i].draw(screen)

    ball.move(ball_position[0], ball_position[1], ball_velo)
    ball.draw(screen)

    # Flip the display
    pygame.display.flip()
    clockobject.tick(5)
    if verbose:
        print("render mode - player positions:", player_positions)
    return player_positions, new_possession


obs, _ = reset()
player_positions, player_velos, ball_or_not, last_possession, ball_position, ball_velo = split_obs(obs)
while True:
    step_reward = [0, 0]


    if timestep == 0:
        total_ball_loc_rewardsa = 0
        total_ball_loc_rewardsb = 0

        clockobject = pygame.time.Clock()
        print("Render mode: Human")
        pygame.init()
        pygame.display.set_caption('Soccer')
        display = pygame.display.set_mode((screen_width_ft, screen_height_ft))
        font = pygame.font.SysFont('Arial_bold', 380)

        screen = pygame.display.set_mode([screen_width_ft, screen_height_ft])




        Players = []
        # Create a Player object
        if verbose:
            print("625 player positions", player_positions)
        for i in range(num_players):
            # print(i)
            new_player = Player(player_positions[i * 2], player_positions[(i * 2) + 1])
            if verbose:
                print("new player:", new_player)
                print("given player positions:", player_positions[i*2], player_positions[i*2 + 1])
            Players.append(new_player)
        # Create ball
        ball = Ball()

    if verbose:
        print("621 player positions", player_positions)

    # Check whether ball is in goal or out of bounds or in goal
    terminated, last_possession = check_ball_position(player_positions, ball_position, last_possession)

    if verbose:
        print("625 player positions", player_positions)
        print("190 check player positions:", player_positions)
        print("192 ball or not", ball_or_not)
        print("213 player velos:", player_velos)
        print("191 last possession:", last_possession)

    ball_or_not, last_possession = get_ball(player_positions, ball_position, last_possession, ball_or_not)


    # Find reward based off of location
    reward = ball_location_reward(ball_position)
    if reward_verbose:
        print("ball_location_reward", reward)
        print("total_ball_loc_reward:", total_ball_loc_rewardsa, total_ball_loc_rewardsb)
    total_ball_loc_rewardsa += reward[0]
    total_ball_loc_rewardsb += reward[1]
    step_reward = [step_reward[i] + reward[i] for i in range(len(step_reward))]


    # Give negative rewards if agent it out of bounds
    out_of_bounds_reward(player_positions)

    # Find new player and ball positions
    ball_position, ball_velo = move_ball(player_positions, ball_or_not, ball_position, ball_velo)
    if verbose:
        print("197 check player positions:", player_positions)
        print("203 ball or not", ball_or_not)
        print("230 player velos", player_velos)


    # Check for collisions
    # TODO: add back?
    # player_positions, player_velos = player_collision(player_positions, player_velos)
    if verbose:
        print("659 player positions", player_positions)

    if verbose:
        print("201 check player positions:", player_positions)
        print("212 ball or not - pass ball:", ball_or_not)


    # Render
    player_positions, ball_or_not = render(player_positions, player_velos, ball_position, ball_velo, timestep, ball_or_not)

    if verbose:
        print("671 player positions", player_positions)
        print("step reward:", torch.tensor(step_reward))
    print("741 last_possession:", last_possession)
    timestep += 1

    if terminated:
        print("End of episode")

