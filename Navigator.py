import time

import numpy as np


class Navigator:
    def __init__(self, grid_matrix: np.ndarray, scale_factor: float):
        """
        Initialize Navigator with a gridworld matrix
        and current position marked with 3.
        """
        self.grid = grid_matrix.copy()
        self.rows, self.cols = self.grid.shape
        start_positions = np.argwhere(self.grid == 3)
        if len(start_positions) != 1:
            raise ValueError(
                "Grid must contain exactly one starting position (value 3)."
            )
        self.current_pos = tuple(start_positions[0])
        self.scale_factor = scale_factor
        print(f"[INIT] Current pos: {self.current_pos}")
        print(f"[SCALE] scale_factor={self.scale_factor:.6f}")

    def _is_valid(self, pos):
        # pos is (row, col) or (y, x)
        x, y = pos
        return (
            0 <= x < self.rows
            and 0 <= y < self.cols
            and self.grid[x][y] in (1, 2)  # 1=navigable, 2=target
        )

    def get_neighbors(self):
        """
        Get valid 8-directional neighbors of the current cell.
        """
        x, y = self.current_pos
        directions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]
        return [
            (x + dx, y + dy)
            for dx, dy in directions
            if self._is_valid((x + dx, y + dy))
        ]

    def get_current_pixel(self) -> tuple[int, int]:
        grid_y, grid_x = self.current_pos
        pixel_x = int(grid_x / self.scale_factor)
        pixel_y = int(grid_y / self.scale_factor)
        return (pixel_x, pixel_y)

    def move(self, direction: str) -> np.ndarray:
        """
        Attempt to move in a cardinal direction. If successful:
        - update grid state: clear old position (set to 1)
        - set new position to 3
        Returns:
            Updated grid state (new NumPy array).
        """
        dir_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        if direction not in dir_map:
            raise ValueError(f"Invalid direction: {direction}")

        dx, dy = dir_map[direction]
        x, y = self.current_pos
        new_pos = (x + dx, y + dy)

        if self._is_valid(new_pos):
            self.grid[x][y] = 1  # Reset old location to navigable
            nx, ny = new_pos
            self.grid[nx][ny] = 3  # Mark new location
            self.current_pos = new_pos
            # print(f"[MOVE] {direction} to {new_pos}")
        else:
            # print(f"[BLOCKED] {direction} from {self.current_pos}")
            pass

        return self.grid.copy()

    def click(self):
        print(f"[CLICK] at {self.current_pos}")

    def wait(self, seconds: float):
        print(f"[WAIT] {seconds:.6f} seconds")
        time.sleep(seconds)

    def get_current_position(self):
        return self.current_pos

    def get_grid_state(self):
        return self.grid.copy()

    def fallback_click(self, backup_target_pixel: tuple[int, int]):
        """
        Optional fallback: use pixel location from GridWorld if in-grid click
        has no effect.
        Args:
            backup_target_pixel: (x, y) screen coordinates from GridWorld
        """
        print(f"[FALLBACK CLICK] at screen pixel {backup_target_pixel}")
        # pyautogui.moveTo(*backup_target_pixel)
        # pyautogui.click()
