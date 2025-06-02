import pygame
import sys

class GridVisualizer:
    def __init__(self, grid_size=(20, 20), scale=40, checkbox_pos=(15, 15)):
        self.grid_width, self.grid_height = grid_size
        self.scale = scale
        self.checkbox_pos = checkbox_pos
        self.path = []

        self.window_width = self.grid_width * self.scale
        self.window_height = self.grid_height * self.scale

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("GridWorld Visualizer")
        self.clock = pygame.time.Clock()

    def update(self, agent_pos):
        self.path.append(agent_pos)
        self._draw(agent_pos)
        pygame.display.flip()
        self._check_events()

    def _draw(self, agent_pos):
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x*self.scale, y*self.scale, self.scale, self.scale)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw checkbox
        cb_x, cb_y = self.checkbox_pos
        checkbox_rect = pygame.Rect(cb_x*self.scale, cb_y*self.scale, self.scale, self.scale)
        pygame.draw.rect(self.screen, (0, 200, 0), checkbox_rect)  # green

        # Draw path
        for (x, y) in self.path:
            pos_rect = pygame.Rect(x*self.scale+10, y*self.scale+10, self.scale-20, self.scale-20)
            pygame.draw.ellipse(self.screen, (100, 100, 255), pos_rect)  # blue trail

        # Draw current position
        x, y = agent_pos
        current_rect = pygame.Rect(x*self.scale+5, y*self.scale+5, self.scale-10, self.scale-10)
        pygame.draw.ellipse(self.screen, (255, 0, 0), current_rect)  # red

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
