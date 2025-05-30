import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui


class GridWorld:
    def __init__(self, grid_size: tuple[int, int] = (100, 100), driver=None):
        """
        Create a GridWorld environment from the current webpage.
        Args:
            grid_size: (height, width) of grid representation (matrix)
            driver: Selenium WebDriver instance (optional)
        """
        self.grid_size = grid_size
        self.driver = driver
        self.screenshot = None
        self.gridworld = None
        self.current_loc = None
        self.target_loc = None
        self.screen_size = pyautogui.size()
        pyautogui.FAILSAFE = False
        pass

    def take_screenshot(self) -> np.ndarray:
        # TODO: selenium method
        img = pyautogui.screenshot()
        self.screenshot = np.array(img)
        return self.screenshot

    def get_current_location(self) -> tuple[int, int]:
        position = pyautogui.position()
        self.current_loc = self.scale_position(position.x, position.y)
        return self.current_loc

    def scale_position(self, pos_x, pos_y):
        screenshot_height, screenshot_width, _ = self.screenshot.shape
        logical_width, logical_height = self.screen_size
        if (screenshot_width, screenshot_height) != (
            logical_width,
            logical_height,
        ):
            scale_x = screenshot_width / logical_width
            scale_y = screenshot_height / logical_height
            pos_scaled = (int(pos_x * scale_x), int(pos_y * scale_y))
        else:
            pos_scaled = (pos_x, pos_y)
        return pos_scaled

    def get_target_location_pag(
        self, target_img_path: str, confidence: float = 0.8
    ) -> tuple[int, int]:
        loc = pyautogui.locateCenterOnScreen(target_img_path, confidence)
        if loc is None:
            raise Exception("Error! Target was not found.")
        self.target_loc = (loc.x, loc.y)
        return self.target_loc

    def get_target_location_ocv(
        self, target_img_path: str, confidence: float = 0.8
    ) -> tuple[int, int]:
        if self.screenshot is None:
            self.take_screenshot()
        screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise Exception("Error! Template was not found:", target_img_path)
        res = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED)
        min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
        if min_v <= confidence:
            h, w = template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            self.target_loc = (center_x, center_y)
            return self.target_loc

    def pixel_to_grid_coords(self, px_x: int, px_y: int) -> tuple[int, int]:
        grid_x = int((px_x / self.screen_size.width) * self.grid_size[1])
        grid_y = int((px_y / self.screen_size.height) * self.grid_size[0])
        grid_x = max(0, min(self.grid_size[1] - 1, grid_x))
        grid_y = max(0, min(self.grid_size[0] - 1, grid_y))
        return (grid_x, grid_y)

    def grid_coords_to_pixel(
        self, grid_x: int, grid_y: int
    ) -> tuple[int, int]:
        px_x = int((grid_x / self.grid_size[1] * self.screen_size.width))
        px_y = int((grid_y / self.grid_size[0] * self.screen_size.height))
        return (px_x, px_y)

    def create_grid(self, obstacle_threshold: int = 50) -> np.ndarray:
        self.take_screenshot()
        screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2GRAY)
        screenshot = cv2.resize(screenshot, self.grid_size)
        self.gridworld = np.where(screenshot < obstacle_threshold, 1, 0)
        return self.gridworld

    def setup_env(self, target_img_path: str) -> tuple:
        self.take_screenshot()
        curr_loc = self.get_current_location()
        target_loc = self.get_target_location_ocv(target_img_path)
        if not target_loc:
            raise Exception("Error! Could not find target.")
        self.create_grid()
        curr_grid_loc = self.pixel_to_grid_coords(curr_loc[0], curr_loc[1])
        target_grid_loc = self.pixel_to_grid_coords(
            target_loc[0], target_loc[1]
        )
        return self.gridworld, curr_grid_loc, target_grid_loc

    def display_gridworld(
        self,
        curr_loc: tuple[int, int],
        target_loc: tuple[int, int],
        path: list = None,
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.gridworld, cmap="gray_r")
        plt.plot(
            curr_loc[0], curr_loc[1], "go", markersize=10, label="Current"
        )
        plt.plot(
            target_loc[0], target_loc[1], "ro", markersize=10, label="Target"
        )
        if path:
            path_x, path_y = zip(*path)
            plt.plot(
                path_x, path_y, "b-", linewidth=2, alpha=0.7, label="Path"
            )
        plt.title("Gridworld")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    import time

    target_img_path = "./data/target.png"
    g1 = GridWorld()
    time.sleep(3)
    g1.setup_env(target_img_path)
    g1.display_gridworld(
        g1.get_current_location(),
        g1.get_target_location_ocv(target_img_path),
    )
    print(f"current mouse position: {g1.current_loc}")
