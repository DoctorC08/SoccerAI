# Import the necessary libraries
import numpy as np

import pygame
import gym
from gym import spaces
from gym.spaces import Box, Tuple, Discrete

import numpy as np

# Initialize pygame
pygame.init()

field_width = 345
field_height = 225
# Set up the drawing window
screen_width_ft = (field_width + 10 + 10) * 2 # 10 feet on each side
screen_height_ft = (field_height + 20 + 20) * 2  # 20 feet on each side
screen = pygame.display.set_mode([screen_width_ft, screen_height_ft])

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

    def move(self, pressed_keys):
        if pressed_keys[pygame.K_UP] and self.y - self.radius > box_y:
            self.y -= 5
            self.action = []
        if pressed_keys[pygame.K_DOWN] and self.y + self.radius < box_y + box_height:
            self.y += 5
        if pressed_keys[pygame.K_LEFT] and self.x - self.radius > box_x:
            self.x -= 5
        if pressed_keys[pygame.K_RIGHT] and self.x + self.radius < box_x + box_width:
            self.x += 5

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 0, 0), (self.x, self.y), self.radius)  # Red circle

# Define the ball for pygame
class Ball:
    def __init__(self):
        self.radius = 3
        self.x = center_x
        self.y = center_y
        self.x_velo = 0
        self.y_velo = 0
    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), self.radius)

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

        # Step function!
        self.step_reward = [0, 0]

        #define timesteps
        self.max_steps = 5_000
        self.current_step = 0

        # define action and obs space
        # Define the action space
        self.action_space = Tuple((
            Discrete(6),
            Discrete(5),
            Discrete(2),
            Discrete(2),
            Discrete(2),
        ))
        # Repeat the action space 11 times
        self.action_space = Tuple([self.action_space for i in range(4)])

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
        player_positions = [[0, 0] for i in range(self.num_players)]
        half_num_players = self.num_players // 2
        # Reset player positions
        for i in range(half_num_players):
            player_positions[i][0] = self.field_bounds_x + (self.field_width // 4)
            player_positions[i][1] = self.field_bounds_y + (self.field_height // 2) + (i * 20)
        for i in range(half_num_players):
            player_positions[i + half_num_players][0] = self.field_bounds_x + (self.field_width // 4) * 3
            player_positions[i + half_num_players][1] = self.field_bounds_y + (self.field_height // 2) + (i * 20)
        player_velos = [[0, 0] for i in range(self.num_players)]


        player_orientations = [0 for i in range(len(player_positions))] # Reset orientation
        player_sprint_bars = [self.total_sprint_bar for i in range(len(player_positions))] # Reset sprint bars
        ball_or_not = [0 for i in range(self.num_players)] # Reset ball or not
        last_possession = [0, 0] # Reset last possession
        ball_position = [self.field_bounds_x + self.field_width // 2, self.field_bounds_y + self.field_height // 2] # Reset ball position
        ball_velo = [0, 0]


        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption('Soccer')
            self.display = pygame.display.set_mode((screen_width_ft, screen_height_ft))
            self.font = pygame.font.SysFont('Arial_bold', 380)

            # Create a Player object
            self.player = Player()
            # Create ball
            self.ball = Ball()

            self.render()

        return [player_positions, player_orientations, player_velos, player_sprint_bars, ball_or_not, last_possession, ball_position, ball_velo]



    # Player orientation will be in radians. 0 radians will be facing to the "right" in the enviornment
    def step(self, obs, player_actions, timestep, render_mode = None):
        # Split obs
        player_positions, player_orientations, player_velos, player_sprint_bars, ball_or_not, last_possession, ball_position, ball_velo = self.split_obs(obs)

        # reset step_reward
        self.step_reward = [0, 0]

        # Check whether ball is in goal or out of bounds or in goal
        terminated, possession, player_positions, player_velos, ball_or_not = self.check_ball_position(player_positions, player_velos, ball_position, last_possession)

        # Find reward based off of location
        reward = self.ball_location_reward(ball_position)
        self.step_reward = [self.step_reward[i] + reward[i] for i in range(len(self.step_reward))]

        # Find new player and ball positions
        player_positions, player_velos, player_orientations, player_sprint_bar = self.player_position(player_positions, player_orientations, player_velos, player_actions, player_sprint_bars)
        ball_position, ball_velo = self.move_ball(player_positions, player_orientations, ball_or_not, ball_position, ball_velo)

        # Check for collisions
        player_positions, player_velos = self.player_collision(player_positions, player_velos)

        # Pass ball
        for i in range(len(ball_or_not)):
            if ball_or_not[i] == 1 and player_actions[i][4] == 1:
                ball_velo = self.pass_ball(player_orientations[i])

        # Check tackles
        ball_or_not = self.get_tackled(player_positions, player_orientations, player_actions)

        # calculate new ball velo based on friction
        ball_velo = [(self.meu_friction * self.ball_weight) + ball_velo[i] for i in range(2)]

        # TODO: offsides?

        # Render
        if self.render_mode == 'human':
            self.render()

        # Check max timesteps
        truncated = self.terminated_or_not(timestep)

        return [player_positions, player_orientations, player_velos, player_sprint_bar, ball_or_not, last_possession, ball_position, ball_velo], self.step_reward, terminated, truncated

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
            return True
        else:
            return False

    # distance function
    def find_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Check whether ball is in goal or out of bounds
    # Last possession should be a 2 length list.
    def check_ball_position(self, player_positions, player_velos, ball_position, last_possession):
        new_ball_position = ball_position
        new_possession = last_possession
        new_player_positions = player_positions
        new_player_velos = player_velos
        new_ball_or_not = [0 for i in range(len(player_positions))]

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
            new_possession = last_possession.reverse() # create new posession

            # Define new x coordinate for ball depending on which side it went out on
            if ball_position[0] < self.field_bounds_x:
                if ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x, self.field_bounds_y]
                elif ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x, self.field_bounds_y + self.field_height]

            elif ball_position > self.field_width + self.field_bounds_x:
                if ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x + self.field_width, self.field_bounds_y]
                elif ball_position[1] < self.field_bounds_y + (self.field_height // 2):
                    new_ball_position = [self.field_bounds_x + self.field_width, self.field_bounds_y + self.field_height]

            # Rewards based on possession
            if new_possession == [1, 0]:
                self.step_reward[0] += .1
                self.step_reward[1] -= .1
            else:
                self.step_reward[0] += .1
                self.step_reward[1] -= .1

            new_player_positions, new_player_velos, new_ball_or_not = self.corner_kick_or_goal_kick(new_possession, player_positions, player_velos, new_ball_position)


        # If ball is out for a throwin
        elif ball_position[1] < self.field_bounds_y or ball_position[1] > self.field_height + self.field_bounds_y:
            new_possession = last_possession.reverse()
            # Create new ball position so ball isn't out of bounds
            if ball_position[1] < self.field_bounds_y: 
                new_ball_position[1] = self.field_bounds_y
            else:
                new_ball_position[1] = self.field_bounds_y + self.field_height
            new_player_positions, new_player_velos, new_ball_or_not = self.throwin(new_possession, player_positions, player_velos, new_ball_position)

            # Rewards based on possession
            if new_possession == [1, 0]:
                self.step_reward[0] += .05
                self.step_reward[1] -= .05
            else:
                self.step_reward[0] += .05
                self.step_reward[1] -= .05
        else:
            new_possession = last_possession



        return terminated, new_possession, new_player_positions, new_player_velos, new_ball_or_not


    # Define throwin function
    # need to define which player should take it
    def throwin(self, new_possession, player_positions, player_velos, ball_position):
        # Find closest player
        closest_player = self.closest_player(new_possession, player_positions, ball_position)

        # Define new player positions and velos and who has the ball
        new_player_positions = player_positions
        new_player_positions[closest_player] = [ball_position[0], ball_position[1]]

        new_player_velos = player_velos
        new_player_velos[closest_player] = [0, 0]

        # Subtract one from ball_or_not list so i don't have to delete a 0
        new_ball_or_not = [0 for i in range(len(player_positions)-1)]
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
        new_player_positions[closest_player] = [new_ball_position[0], new_ball_position[1]]

        new_player_velos = player_velos
        new_player_velos[closest_player] = [0, 0]

        # Subtract one from ball_or_not list so i don't have to delete a 0
        new_ball_or_not = [0 for i in range(len(player_positions) - 1)]
        new_ball_or_not.insert(closest_player, 1)

        return new_player_positions, new_player_velos, new_ball_or_not

    # find closest player to the ball depending on who has possession
    def closest_player(self, new_possession, player_positions, ball_position):
        other_team = len(player_positions) // 2
        if new_possession == [1, 0]:
            for i in player_positions:
                distance = self.find_distance(player_positions[i][0], player_positions[i][1], ball_position[0],
                                              ball_position[1])
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_player = i
        else:
            for i in player_positions:
                distance = self.find_distance(player_positions[i + other_team][0], player_positions[i + other_team][1],
                                              ball_position[0], ball_position[1])
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_player = i + other_team

        return closest_player

    # Calculate closness to goals and return rewards based on that
    def ball_location_reward(self, ball_position):
        if ball_position[0] > (self.field_width/2) + self.field_bounds_x:
            # distance formula stuff and fraction of that for reward
            distance = self.find_distance(ball_position[0], ball_position[1], self.center_goal_position_x2, self.center_goal_position_y)
            if 1 / (2 * distance) < .5:
                return [1 / (2 * distance), -1 / (2 * distance)]
            else:
                return [0.5, -0.5]
        else:
            distance = self.find_distance(ball_position[0], ball_position[1], self.center_goal_position_x1, self.center_goal_position_y)
            if 1 / (2 * distance) < .5:
                return [-1 / (2 * distance), 1 / (2 * distance)]
            else:
                return [-0.5, 0.5]

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
            new_ball_or_not = [0 for i in range(len(player_positions) - 1)]
            try:
                new_ball_or_not.insert(player_positions.index(player_position), 1) # find index of player inside of player_positions

                # Check which team has possession
                if player_positions.index(player_position) > len(player_positions) // 2:
                    last_possession = [0, 1]
                else:
                    last_possession = [1, 0]
            except:
                new_ball_or_not = [0 for i in range(len(player_positions))] # This should not be reached
                print("------------------------------")
                print("--- ERROR IN get_ball func ---")
                print("------------------------------")

            return new_ball_or_not, last_possession

    # Get tackled if enemy is close and tackles and player has ball
    # player_tackled_or_not should be a whole list of 0s and 1s determining if they decided to or not
    def get_tackled(self, player_positions, player_orientation, player_actions):
        # Initialize an empty list to store the new possession
        new_possession = [0] * len(player_positions)

        # Loop through each player
        for i in range(len(player_positions)):
            # Check if the player wants to tackle
            if player_actions[i][3]:
                # Get the player's orientation and position
                player1_orientation = player_orientation[i]
                player1_position = player_positions[i]

                # Loop through the other players
                for j in range(len(player_positions)):
                    # Check if the other player has the ball
                    if new_possession[j]:
                        # Get the other player's orientation and position
                        player2_orientation = player_orientation[j]
                        player2_position = player_positions[j]

                        # Check if the two players are facing each other and are within 5
                        distance, facing_each_other = self.check_orientation(player1_orientation, player1_position, player2_orientation, player2_position)
                        if distance <= 5 and facing_each_other:
                            # The player has tackled the ball
                            new_possession[i] = 1
                            new_possession[j] = 0

                            # Give rewards to player who tackled successfully
                            if player_positions.index(player1_orientation) > len(player_positions) // 2:
                                self.step_reward[0] -= .1
                                self.step_reward[1] += .1
                            else:
                                self.step_reward[0] += .1
                                self.step_reward[1] -= .1
                        else:
                            # If player is on second team
                            if player_positions.index(player1_orientation) > len(player_positions) // 2:
                                self.step_reward[0] += .1
                                self.step_reward[1] -= .1
                            else:
                                self.step_reward[0] -= .1
                                self.step_reward[1] += .1
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
        for i in range(len(player_positions)):
            # Find new player velo and orientation
            player_action_velo, new_orientation = self.generate_velocity_vector_and_orientation(player_actions[i], player_velos[i], player_orientations[i])

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
        new_pos = [player_positions[i] + player_velos[i] for i in range(len(player_positions))]

        return new_pos, player_action_velos, new_orientations, new_player_sprint_bars

    def generate_velocity_vector_and_orientation(self, player_action, player_velo, player_orientation):
        # Define the angles in radians
        angles_in_radians = [np.pi / 6, np.pi / 12, 0, -np.pi / 12, -np.pi / 6, np.pi]

        # Find the angle based on the input index
        angle = angles_in_radians[player_action[0]]

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

        final_velo = np.round(velocity, 5) + player_velo
        mag_final_velo = self.find_distance(velocity[0], velocity[1], 0, 0)
        if mag_final_velo > self.player_max_speed:
            # Normalize and then multiply by max_speed
            final_velo[0] = (final_velo[0] // mag_final_velo) * self.player_max_speed
            final_velo[1] = (final_velo[1] // mag_final_velo) * self.player_max_speed


        return final_velo, new_orientation

    # Move ball depending on players and ball velo if no one has it
    # Player positions should be new positions so we can move ball to player
    def move_ball(self, player_positions, player_orientations, ball_or_not, ball_position, ball_velo):
        new_ball_position = [0, 0]
        # Check if someone has possession
        if any(element == 1 for element in ball_or_not):
            # Find index of possession
            for i in range(len(ball_or_not)):
                if ball_or_not[i]:
                    index = i

            # Calculate ball position based on player position and orientation of player
            new_ball_position[0] = player_positions[i][0] + 3 * np.cos(player_orientations[i][0])
            new_ball_position[1] = player_positions[i][1] + 3 * np.cos(player_orientations[i][1])

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
        other_team = len(player_positions) // 2
        for i in range(len(player_positions) // 2):
            for j in range(len(player_positions) // 2):
                # Find distance between players
                distance = self.find_distance(player_positions[i][0], player_positions[i][1], player_positions[j + other_team][0], player_positions[j + other_team][1])
                if distance < 3:
                    # find magnitude for vectors through distance formulas
                    normalize = self.find_distance(player_velos[i][0], player_velos[i][1], 0, 0)

                    # Calculat new player positions and velos
                    new_player_positions[i] == [player_positions[i][0] + (player_velos[i][0]/normalize)]
                    new_player_positions[j + other_team] == [player_positions[j + other_team][0] + (player_velos[j + other_team][0]/normalize)]

                    new_player_velos[i] == [0, 0]
                    new_player_velos[j + other_team] = [0, 0]

        return new_player_positions, new_player_velos

    # Offsides function
    def offsides(self, player_positions, ball_or_not):
        # Get the player who has the ball
        ball_holder_index = ball_or_not.index(1)
        # ball_holder_position = player_positions[ball_holder_index]

        # Get the positions of all of the players on the opposing team and find the last 2 defenders
        other_team = len(player_positions) // 2
        if ball_holder_index > other_team:
            opposing_team_positions = [player_positions[i] for i in range(other_team)]

            # Find last 2 defenders
            last_defender = player_positions[other_team][0]
            second_last_defender = player_positions[other_team + 1][0]
            for i in range(other_team - 2):
                if player_positions[i + 2 + other_team][0] < second_last_defender:
                    if player_positions[i + 2 + other_team][0] < last_defender:
                        second_last_defender = last_defender
                        last_defender = player_positions[i + 2 + other_team][0]
                    else:
                        second_last_defender = player_positions[i + 2 + other_team][0]

        else:
            opposing_team_positions = [player_positions[i + other_team] for i in range(other_team)]

            # Find last 2 defenders
            last_defender = player_positions[0][0]
            second_last_defender = player_positions[1][0]
            for i in range(other_team - 2):
                if player_positions[i + 2][0] < second_last_defender:
                    if player_positions[i + 2][0] < last_defender:
                        second_last_defender = last_defender
                        last_defender = player_positions[i + 2][0]
                    else:
                        second_last_defender = player_positions[i + 2][0]

        # Check if any of the players on the opposing team are offside
        offside_players = []
        for opposing_team_player_position in opposing_team_positions:
            for i in range(other_team):
                # Check if the opposing team player is closer to the ball than the second-to-last defender (i.e. the goalkeeper)
                if opposing_team_player_position[0] < second_last_defender:
                    self.step_reward[0] -= .2
                    self.step_reward[1] += .2

    # Split obs
    def split_obs(self, obs):
        print("Observations:", obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7])
        return obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7]

    def render(self, mode='human'):
        # TODO: Fix drawing players and controls and what not.... prob after i define nns
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
        pressed_keys = pygame.key.get_pressed()
        self.player.move(pressed_keys)
        self.player.draw(screen)

        self.ball.draw(screen)

        # Flip the display
        pygame.display.flip()



    def close(self):
        pygame.quit()

Env = env()
obs = Env.reset()
actions = Env.action_space.sample() # Sample action for action space
max_timestep = 5_000
print(actions)
obs, reward, terminated, truncated = Env.step(obs, actions, 5000)
print("New Observations:", obs)
print(reward)
print(terminated, truncated)
