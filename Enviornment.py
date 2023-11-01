# Import the necessary libraries
import pygame

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

# Define the Player class
class Player:
    def __init__(self):
        self.radius = 8  # Radius for the circle
        self.x = center_x
        self.y = center_y

    def move(self, pressed_keys):
        if pressed_keys[pygame.K_UP] and self.y - self.radius > box_y:
            self.y -= 5
        if pressed_keys[pygame.K_DOWN] and self.y + self.radius < box_y + box_height:
            self.y += 5
        if pressed_keys[pygame.K_LEFT] and self.x - self.radius > box_x:
            self.x -= 5
        if pressed_keys[pygame.K_RIGHT] and self.x + self.radius < box_x + box_width:
            self.x += 5

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 0, 0), (self.x, self.y), self.radius)  # Red circle

class Ball:
    def __init__(self):
        self.radius = 3
        self.x = center_x
        self.y = center_y
        self.x_velo = 0
        self.y_velo = 0
    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), self.radius)





# Create a Player object
player = Player()
# Create ball
ball = Ball()

# Run until the user asks to quit
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
    pygame.draw.rect(screen, (0, 0, 0), (center_x - 1, center_y - (box_height/2), 2, box_height))
    # Draw goals
    # Goal #1:
    pygame.draw.rect(screen, (0, 0, 0), (box_x - 20, box_y + box_height//2 - 24, box_x, box_y))
    # Goal #2:
    pygame.draw.rect(screen, (0, 0, 0), (box_x + box_width, box_y + box_height//2 - 24, box_x + box_width, box_y))



    # Move and draw the player
    pressed_keys = pygame.key.get_pressed()
    player.move(pressed_keys)
    player.draw(screen)

    ball.draw(screen)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
