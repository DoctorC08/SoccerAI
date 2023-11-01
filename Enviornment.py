# Enviornment.py
# by christophermao
# 10/31/23
import pygame
pygame.init()

# Set up the drawing window
screen_width_ft = (345 + 10 + 10) * 2  # 10 feet on each side
screen_height_ft = (225 + 20 + 20) * 2  # 20 feet on each side
screen = pygame.display.set_mode([screen_width_ft, screen_height_ft])

# Goals are 24 feet long

# Import controls for testing
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_w,
    K_s,
    K_a,
    K_d,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# Define a Player object by extending pygame.sprite.Sprite
# The surface drawn on the screen is now an attribute of 'player'
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((75, 75))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()

    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -5)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 5)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-5, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(5, 0)

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > screen_width_ft:
            self.rect.right = screen_width_ft
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= screen_height_ft:
            self.rect.bottom = screen_height_ft

player = Player()

# Run until the user asks to quit
running = True
while running:

    # Close window function
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Player movement
    pressed_keys = pygame.key.get_pressed()
    player.update(pressed_keys)

    # Fill the background with green
    screen.fill((50, 168, 82))

    # Draw player
    screen.blit(player.surf, player.rect.topleft)  # Use player's rect to determine position

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
