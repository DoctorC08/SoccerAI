import pygame
import numpy as np

class SoccerRenderer: 
    def __init__(self, h: int, w: int, ui_width: int, display_mode, fps: int = 30):
        self.h = h
        self.w = w
        self.ui_width = ui_width
        self.fps = fps
        self.display_mode = display_mode

        pygame.init()
        if display_mode == "human":
            self.screen = pygame.display.set_mode((self.w + self.ui_width, self.h))
        else: 
            self.screen = pygame.Surface((self.w + self.ui_width, self.h))
        pygame.display.set_caption("Soccer Environment")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.reward_history = []
    
    
    def draw(self, state, reward, timer_info=None, action_info=None):
        self.screen.fill((30, 30, 30)) 
        
        entities = np.reshape(state, (-1, 4)) # x, y, vx, vy
        
        # Draw field
        field_rect = pygame.Rect(0, 0, self.w, self.h)
        pygame.draw.rect(self.screen, (34, 139, 34), field_rect) 
        pygame.draw.line(self.screen, (255, 255, 255), (self.w/2, 0), (self.w/2, self.h), 2)

        for i, ent in enumerate(entities):
            # Scale normalized coordinates back to pixels
            px = int(ent[0] * self.w)
            py_pos = int(ent[1] * self.h)
            color = (255, 255, 255) if i == 0 else (255, 0, 0) 
            
            pygame.draw.circle(self.screen, color, (px, py_pos), 10)
            
            # Draw Velocity Vectors
            pygame.draw.line(self.screen, (255, 255, 0), (px, py_pos), 
                             (px + ent[2]*20, py_pos + ent[3]*20), 2)

        # Draw stats panel
        stats_rect = pygame.Rect(self.w, 0, self.ui_width, self.h)
        pygame.draw.rect(self.screen, (50, 50, 50), stats_rect)
        self.reward_history.append(reward)

        self._draw_hud(stats_rect, timer_info=timer_info, action_info=action_info)
        
        # Draw line graph for rewards
        if self.reward_history and len(self.reward_history) > 1:
            self._draw_graph(self.reward_history, stats_rect)

        if self.display_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            action = 0 # Default: Stay
            keys = pygame.key.get_pressed()
            if keys[pygame.K_e]: action = 5 # Kick
            if keys[pygame.K_w]: action = 4 # North
            if keys[pygame.K_s]: action = 3 # South
            if keys[pygame.K_a]: action = 2 # West
            if keys[pygame.K_d]: action = 1 # East
            self.clock.tick(self.fps)

            pygame.display.flip()
            return [action]
            
        # Return frame for W&B
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _draw_hud(self, rect, timer_info=None, action_info=None):
        y = rect.y + 10
        text_color = (235, 235, 235)

        if timer_info is not None:
            step = int(timer_info.get("step", 0))
            max_steps = int(timer_info.get("max_steps", 0))
            remaining_steps = int(timer_info.get("remaining_steps", 0))
            remaining_seconds = float(timer_info.get("remaining_seconds", 0.0))

            step_text = self.font.render(
                f"Step: {step}/{max_steps}  Left: {remaining_steps}",
                True,
                text_color,
            )
            self.screen.blit(step_text, (rect.x + 8, y))
            y += 22

            timer_text = self.font.render(f"Time Left: {remaining_seconds:.1f}s", True, text_color)
            self.screen.blit(timer_text, (rect.x + 8, y))
            y += 28

        if action_info is not None:
            action_header = self.font.render("Recent Actions", True, (200, 255, 200))
            self.screen.blit(action_header, (rect.x + 8, y))
            y += 22

            last_actions = action_info.get("last_actions", [])
            if last_actions:
                latest_text = self.font.render(f"Latest: {last_actions}", True, text_color)
                self.screen.blit(latest_text, (rect.x + 8, y))
                y += 20

            for step_idx, step_actions in reversed(action_info.get("history", [])):
                history_text = self.font.render(f"t={step_idx}: {step_actions}", True, (190, 190, 190))
                self.screen.blit(history_text, (rect.x + 8, y))
                y += 18

                if y > rect.centery - 8:
                    break

    def _draw_graph(self, data, rect):
        if not data or len(data) < 2: return
        
        # Show last n steps
        display_data = np.array(data[-200:]) 
        num_players = display_data.shape[1] if len(display_data.shape) > 1 else 1
        
        # Cyan for P1, Orange for P2
        colors = [(0, 255, 255), (255, 100, 0)] 
        
        # Scaling factor
        data_max = 1
        scale_y = 50 / data_max if data_max != 0 else 1

        for p_idx in range(num_players):
            points = []
            player_rewards = display_data[:, p_idx]
            
            for i, val in enumerate(player_rewards):
                # Calculate X: spread across the UI width
                x = rect.x + (i * (rect.width / (len(player_rewards) - 1)))
                # Calculate Y: centered in the panel
                y = rect.centery - (val * scale_y)
                points.append((x, y))
                
            if len(points) > 1:
                pygame.draw.lines(self.screen, colors[p_idx % len(colors)], False, points, 2)
                
                label = self.font.render(f"P{p_idx}: {player_rewards[-1]:.2f}", True, colors[p_idx % len(colors)])
                self.screen.blit(label, (rect.x + 5, rect.y + 10 + (p_idx * 20)))