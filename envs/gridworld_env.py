import gymnasium
import numpy as np
from gymnasium import spaces

import pag_nav as pag
from GridWorld import GridWorld
from Navigator import Navigator


class ScreenGridWorldEnv(gymnasium.Env):
    """
    A custom Gym environment that wraps the GridWorld and Navigator classes
    to provide an RL interface for screen interaction.
    """

    metadata = {"render_modes": ["console"], "render_fps": 4}

    def __init__(
        self,
        target_img_path: str,
        matrix_size: int = 1000,
        max_steps=1000,
        render_mode=None,
    ):
        """
        Initializes the ScreenGridWorldEnv.

        Args:
            target_img_path (str): Path to the target image for template matching.
            matrix_size (int): The size of the square grid (e.g., 1000 for a 1000x1000 grid).
            render_mode (str, optional): The rendering mode. Currently only "console"
                                         is supported for printing grid state.
        """
        super().__init__()

        self.target_img_path = target_img_path
        self.matrix_size = matrix_size

        self.grid_world = GridWorld(matrix_size=self.matrix_size)
        self.navigator = None  # Will be initialized in reset()

        # Define observation space based on the grid_world's matrix_size
        # Agent and target locations are (x, y) grid coordinates
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0, self.matrix_size - 1, shape=(2,), dtype=int
                ),
                "target": spaces.Box(
                    0, self.matrix_size - 1, shape=(2,), dtype=int
                ),
            }
        )

        self.action_space = spaces.Discrete(5)

        # Mapping Gym action (int) to Navigator direction (string)
        # Note: Navigator's move expects (dy, dx) for (row, col) indexing.
        # Our Gym agent_location is (x, y).
        # So, Gym action 0 (Right) means +1 to x, 0 to y.
        # Navigator's "right" means +1 to col (x), 0 to row (y). This matches.
        # Gym action 1 (Up) means 0 to x, -1 to y.
        # Navigator's "up" means -1 to row (y), 0 to col (x). This matches.
        self._gym_action_to_navigator_direction = {
            0: "right",
            1: "up",
            2: "left",
            3: "down",
            4: "wait",  # Added wait action
        }

        assert (
            render_mode is None
            or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

        # Agent and target locations in Gym's (x, y) format
        self._agent_location = None
        self._target_location = None
        # Stores sequence of (x, y, action_taken_to_reach_this_state) grid coordinates
        self._agent_path = []
        self.max_steps = max_steps
        self.step_count = 0

    def _get_obs(self):
        """
        Returns the current observation of the environment.
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        """
        Returns additional information about the environment state.
        """
        # Using Manhattan distance (L1 norm) as a simple metric
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _grid_to_screen_center_pixel(
        self, grid_x: int, grid_y: int
    ) -> tuple[int, int]:
        """
        Translates grid coordinates (x, y) to the center pixel (x, y) on the original screen.
        """
        # Calculate the size of one grid cell in original screen pixels
        original_cell_width = (
            self.grid_world.screenshot_dim[0] / self.grid_world.matrix_size
        )
        original_cell_height = (
            self.grid_world.screenshot_dim[1] / self.grid_world.matrix_size
        )

        # Calculate the top-left pixel of the given grid cell in original screen pixels
        top_left_pixel_x = grid_x * original_cell_width
        top_left_pixel_y = grid_y * original_cell_height

        # Calculate the center of the cell in original screen pixels
        center_pixel_x = int(top_left_pixel_x + original_cell_width / 2)
        center_pixel_y = int(top_left_pixel_y + original_cell_height / 2)

        return center_pixel_x, center_pixel_y

    def _check_click_in_target_region(
        self, click_x: int, click_y: int
    ) -> bool:
        """
        Checks if the given click pixel coordinates fall within the detected target region.
        """
        target_x, target_y = self.grid_world.target_loc
        target_w, target_h = self.grid_world.target_dim

        is_x_in_target = target_x <= click_x <= (target_x + target_w)
        is_y_in_target = target_y <= click_y <= (target_y + target_h)

        return is_x_in_target and is_y_in_target

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state by performing browser setup,
        rebuilding the GridWorld, and initializing the Navigator.
        """
        super().reset(seed=seed)  # Seed the random number generator if needed
        self.step_count = 0

        print("\n--- Resetting Environment: Browser Setup ---")
        pag.browser_setup()
        print(
            "Browser: Browser is now in initial state for training episode."
        )
        print("\n--- Resetting Environment: Building GridWorld ---")
        self.grid_world.build(
            target_img_path=self.target_img_path, save_csv=True
        )
        print("GridWorld built successfully.")
        print(f"GridWorld matrix_size: {self.grid_world.matrix_size}")
        print(
            f"GridWorld mouse_loc (scaled x, y): {self.grid_world.mouse_loc}"
        )
        print(
            f"GridWorld target_loc (original x, y): {self.grid_world.target_loc}"
        )
        print(
            f"GridWorld target_dim (original w, h): {self.grid_world.target_dim}"
        )
        print(f"GridWorld scale_factor: {self.grid_world.scale_factor}")
        # Initialize Navigator with the newly built grid and scale factor
        self.navigator = Navigator(
            self.grid_world.grid_matrix, self.grid_world.scale_factor
        )
        # Navigator's get_current_position() returns (y, x), convert to Gym's (x, y).
        nav_initial_y, nav_initial_x = self.navigator.get_current_position()
        self._agent_location = np.array(
            [nav_initial_x, nav_initial_y], dtype=int
        )
        # Find where '2' (target) is in the grid matrix
        target_grid_coords = np.argwhere(self.grid_world.grid_matrix == 2)
        if len(target_grid_coords) == 0:
            raise ValueError(
                "Target (value 2) not found in the grid matrix after build."
            )
        # target_grid_coords is (y, x), convert to (x, y) for Gym env's _target_location
        self._target_location = np.array(
            [target_grid_coords[0][1], target_grid_coords[0][0]], dtype=int
        )
        # Clear the path for a new episode
        self._agent_path = []
        # Add initial position to path with a 'None' action as it's the starting point
        self._agent_path.append(tuple(self._agent_location) + (None,))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "console":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Takes an action in the environment, performs the move in the
        gridworld. If the target is reached, executes the full path
        with pyautogui and clicks.
        """
        self.step_count += 1
        # Translate Gym action to Navigator direction string
        direction_str = self._gym_action_to_navigator_direction[action]

        # Get current agent grid location (x, y) before potential move
        current_agent_grid_x, current_agent_grid_y = self._agent_location

        if direction_str == "wait":
            # If it's a wait action, the agent's grid position does not change.
            # The new_agent_location is effectively the same as the old one.
            new_agent_location = self._agent_location.copy()
        else:
            # For movement actions, perform the move using the Navigator.
            # Navigator updates its internal grid and current_pos (y, x).
            self.navigator.move(direction_str)
            # Get the new agent location from Navigator (y, x) and convert to Gym's (x, y)
            nav_new_y, nav_new_x = self.navigator.get_current_position()
            new_agent_location = np.array([nav_new_x, nav_new_y], dtype=int)

        # Update Gym env's agent location
        self._agent_location = new_agent_location
        # Add current position and the action that led to it to the path
        self._agent_path.append(tuple(self._agent_location) + (action,))

        # Check termination condition
        agent_grid_y, agent_grid_x = (
            self._agent_location[1],
            self._agent_location[0],
        )
        terminated = (
            self.grid_world.grid_matrix[agent_grid_y, agent_grid_x] == 2
        )
        truncated = self.step_count >= self.max_steps
        reward = 0  # Default reward

        if terminated:
            print(
                "\n--- Agent reached target in gridworld! Executing full path on screen ---"
            )
            # for i, path_point in enumerate(self._agent_path):
            #     grid_x, grid_y, action_taken_at_this_point = path_point

            #     if (
            #         action_taken_at_this_point == 4
            #     ):  # If this point was reached by a 'wait' action
            #         print(f"  Performing wait at grid ({grid_x}, {grid_y})")
            #         pag.idle()  # Use the idle function from pag.py
            #     else:
            #         # For initial point (None action) or a movement action
            #         pyautogui_x, pyautogui_y = (
            #             self._grid_to_screen_center_pixel(grid_x, grid_y)
            #         )
            #         print(
            #             f"  Moving mouse to grid ({grid_x}, {grid_y}) -> screen ({pyautogui_x}, {pyautogui_y})"
            #         )
            #         pag.move_mouse(pyautogui_x, pyautogui_y)

            # After all moves, perform the click at the final target location
            final_click_x, final_click_y = self._grid_to_screen_center_pixel(
                self._agent_location[0], self._agent_location[1]
            )
            print(
                f"  Performing click at final screen position: ({final_click_x}, {final_click_y})"
            )
            # pag.mouse_click(final_click_x, final_click_y)

            # Determine reward based on whether the click was in the target region
            if self._check_click_in_target_region(
                final_click_x, final_click_y
            ):
                reward = 1
                print(
                    f"  Click successful! Final click ({final_click_x}, {final_click_y}) was inside target region."
                )
            else:
                reward = 0
                print(
                    f"  Click missed! Final click ({final_click_x}, {final_click_y}) was outside target region."
                )
            print(f"  Reward for this episode: {reward}")

            # Perform browser teardown after the episode is complete
            print("\n--- Episode Complete: Browser Teardown ---")
            pag.browser_teardown()
            print("Browser: Browser state reset for next episode.")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "console":
            self._render_frame()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """
        Renders the environment. For this environment, it prints the grid state to the console.
        """
        if self.render_mode == "console":
            self._render_frame()

    def _render_frame(self):
        """
        Internal method to render the current frame to the console.
        Uses the Navigator's current grid state.
        """
        if self.render_mode == "console" and self.navigator:
            grid = self.navigator.get_grid_state()
            # Note: Navigator's grid is (y, x), and its current_pos is (y, x)
            # The agent and target locations in Gym env are (x, y).
            # For console display, we need to use the Navigator's grid directly.

            # Determine a reasonable 'stride' for printing large grids to avoid excessively long lines
            stride = max(
                1, self.matrix_size // 50
            )  # Print every 50th cell if matrix_size is large

            print("\n" + "=" * (self.matrix_size * 2 // stride + 1))
            for r_idx in range(0, self.matrix_size, stride):
                row_str = []
                for c_idx in range(0, self.matrix_size, stride):
                    cell_val = grid[r_idx, c_idx]
                    if (
                        r_idx,
                        c_idx,
                    ) == self.navigator.get_current_position():
                        row_str.append("A")  # Agent
                    elif cell_val == 2:
                        row_str.append("T")  # Target
                    elif cell_val == 1:
                        row_str.append(".")  # Navigable
                    else:
                        row_str.append("#")  # Non-navigable (0)
                print("|" + "|".join(row_str) + "|")
            print("=" * (self.matrix_size * 2 // stride + 1))
            print(
                f"Agent Grid (y, x): {self.navigator.get_current_position()}"
            )
            # Note: _target_location in Gym env is (x, y) based on GridWorld.target_loc (original pixels)
            # For console output, it might be more intuitive to show its grid equivalent.
            target_grid_x = self._target_location[0]
            target_grid_y = self._target_location[1]
            print(f"Target Grid (x, y): ({target_grid_x}, {target_grid_y})")
            print(f"Distance: {self._get_info()['distance']}")

    def close(self):
        """
        Performs any necessary cleanup.
        """
        print("Closing ScreenGridWorldEnv.")
        # No specific resources to close for GridWorld/Navigator beyond Python's GC
