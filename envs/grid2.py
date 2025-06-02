import gymnasium
import numpy as np
from gymnasium import spaces

import pag_nav as pag
from GridWorld import GridWorld
from Navigator import Navigator


class DiscreteGridWorldEnv(gymnasium.Env):
    metadata = {"render_modes": ["console"], "render_fps": 4}

    def __init__(
        self,
        target_img_path: str,
        matrix_size: int = 1000,
        max_steps: int = 1000,
        render_mode: str = None,
        is_eval_mode: bool = False,
    ):
        super().__init__()
        self.target_img_path = target_img_path
        self.matrix_size = matrix_size
        self.max_steps = max_steps
        self.is_eval_mode = is_eval_mode
        self.grid_world = GridWorld(matrix_size=self.matrix_size)
        self.navigator = None
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
        self._gym_action_to_navigator_direction = {
            0: "right",
            1: "up",
            2: "left",
            3: "down",
            4: "wait",
        }
        assert (
            render_mode is None
            or render_mode in self.metadata["render_modes"]
        ), (
            f"Render mode '{render_mode}' not supported. Must be one of {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode
        self._agent_location = None
        self._target_location = None
        self._agent_path = []
        self.step_count = 0

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        agent_pixel_x, agent_pixel_y = self._grid_to_screen_center_pixel(
            self._agent_location[0], self._agent_location[1]
        )
        target_pixel_x, target_pixel_y = self.grid_world.target_loc

        pixel_distance = np.linalg.norm(
            np.array([agent_pixel_x, agent_pixel_y])
            - np.array([target_pixel_x, target_pixel_y]),
            ord=2,
        )

        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "pixel_distance_to_actual_target": pixel_distance,
        }

    def _grid_to_screen_center_pixel(
        self, grid_x: int, grid_y: int
    ) -> tuple[int, int]:
        original_cell_width = (
            self.grid_world.screenshot_dim[0] / self.grid_world.matrix_size
        )
        original_cell_height = (
            self.grid_world.screenshot_dim[1] / self.grid_world.matrix_size
        )
        top_left_pixel_x = grid_x * original_cell_width
        top_left_pixel_y = grid_y * original_cell_height
        center_pixel_x = int(top_left_pixel_x + original_cell_width / 2)
        center_pixel_y = int(top_left_pixel_y + original_cell_height / 2)
        return center_pixel_x, center_pixel_y

    def _check_click_in_target_region(
        self, click_x: int, click_y: int
    ) -> bool:
        target_x, target_y = self.grid_world.target_loc
        target_w, target_h = self.grid_world.target_dim
        is_x_in_target = target_x <= click_x <= (target_x + target_w)
        is_y_in_target = target_y <= click_y <= (target_y + target_h)
        return is_x_in_target and is_y_in_target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        print("\n--- Resetting Environment ---")
        print("Browser: Performing browser setup.")
        pag.browser_setup()
        print("Browser: Browser is now in initial state.")
        print("GridWorld: Building GridWorld from current screen state.")
        self.grid_world.build(
            target_img_path=self.target_img_path, save_csv=True
        )
        print("GridWorld: GridWorld built successfully.")
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
        self.navigator = Navigator(
            self.grid_world.grid_matrix, self.grid_world.scale_factor
        )
        nav_initial_y, nav_initial_x = self.navigator.get_current_position()
        self._agent_location = np.array(
            [nav_initial_x, nav_initial_y], dtype=int
        )
        target_grid_coords = np.argwhere(self.grid_world.grid_matrix == 2)
        if len(target_grid_coords) == 0:
            raise ValueError(
                "Target (value 2) not found in the grid matrix after build. "
                "Ensure your target image is correctly detected."
            )
        self._target_location = np.array(
            [target_grid_coords[0][1], target_grid_coords[0][0]], dtype=int
        )
        self._agent_path = []
        self._agent_path.append(tuple(self._agent_location) + (None,))
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "console":
            self._render_frame()
        return observation, info

    def step(self, action):
        self.step_count += 1
        direction_str = self._gym_action_to_navigator_direction[action]
        if direction_str == "wait":
            new_agent_location = self._agent_location.copy()
        else:
            self.navigator.move(direction_str)
            nav_new_y, nav_new_x = self.navigator.get_current_position()
            new_agent_location = np.array([nav_new_x, nav_new_y], dtype=int)
        self._agent_location = new_agent_location
        self._agent_path.append(tuple(self._agent_location) + (action,))
        agent_grid_y, agent_grid_x = (
            self._agent_location[1],
            self._agent_location[0],
        )
        terminated = (
            self.grid_world.grid_matrix[agent_grid_y, agent_grid_x] == 2
        )
        truncated = self.step_count >= self.max_steps
        # truncated = self.step_count >= self.max_steps
        reward = 0

        if terminated:
            print("\n--- Agent reached target in gridworld! ---")
            if self.is_eval_mode:
                print(
                    "EVAL MODE: Replaying full path on screen and performing final click."
                )
                for i, path_point in enumerate(self._agent_path):
                    grid_x, grid_y, action_taken_at_this_point = path_point
                    if action_taken_at_this_point is None and i == 0:
                        continue
                    if action_taken_at_this_point == 4:
                        print(
                            f" Performing wait at grid ({grid_x}, {grid_y})"
                        )
                        pag.idle()
                    else:
                        pyautogui_x, pyautogui_y = (
                            self._grid_to_screen_center_pixel(grid_x, grid_y)
                        )
                        print(
                            f" Moving mouse to grid ({grid_x}, {grid_y}) -> screen ({pyautogui_x}, {pyautogui_y})"
                        )
                        pag.move_mouse(pyautogui_x, pyautogui_y)
                final_click_x, final_click_y = (
                    self._grid_to_screen_center_pixel(
                        self._agent_location[0], self._agent_location[1]
                    )
                )
                print(
                    f" Performing click at final screen position: ({final_click_x}, {final_click_y})"
                )
                pag.mouse_click(final_click_x, final_click_y)
                if self._check_click_in_target_region(
                    final_click_x, final_click_y
                ):
                    reward = 1
                    print(
                        f" Click successful! Final click ({final_click_x}, {final_click_y}) was inside target region."
                    )
                else:
                    reward = 0
                    print(
                        f" Click missed! Final click ({final_click_x}, {final_click_y}) was outside target region."
                    )
                print(f" Reward for this episode: {reward}")
            else:
                reward = 1
                print(
                    f"TRAINING MODE: Target reached in gridworld. Reward: {reward}"
                )
                print(f" Reward for this episode: {reward}")
        elif truncated:
            print(
                f"\n--- Episode truncated after {self.step_count} steps. ---"
            )
            reward = 0

        if terminated or truncated:
            print("\n--- Episode Complete: Browser Teardown ---")
            pag.browser_teardown()
            print("Browser: Browser state reset for next episode.")

        observation = self._get_obs()
        info = self._get_info()
        print(f"$$$ path actions: {len(self._agent_path)}")
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
        if self.render_mode == "console":
            self._render_frame()

    def _render_frame(self):
        if self.render_mode == "console" and self.navigator:
            grid = self.navigator.get_grid_state()
            stride = max(1, self.matrix_size // 50)
            print("\n" + "=" * (self.matrix_size * 2 // stride + 1))
            for r_idx in range(0, self.matrix_size, stride):
                row_str = []
                for c_idx in range(0, self.matrix_size, stride):
                    cell_val = grid[r_idx, c_idx]
                    if (
                        r_idx,
                        c_idx,
                    ) == self.navigator.get_current_position():
                        row_str.append("A")
                    elif cell_val == 2:
                        row_str.append("T")
                    elif cell_val == 1:
                        row_str.append(".")
                    else:
                        row_str.append("#")
                print("|" + "|".join(row_str) + "|")
            print("=" * (self.matrix_size * 2 // stride + 1))
            print(
                f"Agent Grid (y, x): {self.navigator.get_current_position()}"
            )
            target_grid_x = self._target_location[0]
            target_grid_y = self._target_location[1]
            print(f"Target Grid (x, y): ({target_grid_x}, {target_grid_y})")
            print(f"Distance: {self._get_info()['distance']}")

    def close(self):
        print("Closing DiscreteGridWorldEnv.")
        print("Browser: Performing final browser teardown.")
        pag.browser_teardown()
