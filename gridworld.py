import cv2
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
        self.start_loc = None
        self.target_loc = None
        self.screen_size = pyautogui.size()
        pyautogui.FAILSAFE = False
        pass

    def take_screenshot(self) -> np.ndarray:
        # TODO: selenium method
        img = pyautogui.screenshot()
        self.screenshot = np.array(img)
        return self.screenshot

    def get_start_location(self) -> tuple[int, int]:
        self.start_loc = pyautogui.position()
        return self.start_loc

    def get_target_location_pag(
        self, target_img_path: str, confidence: float = 0.80
    ) -> tuple[int, int]:
        pass

    def get_target_location_ocv(
        self, target_img_path: str, confidence: float = 0.80
    ) -> tuple[int, int]:
        template = cv2.imread(target_img_path)
        template = cv2.cvtColor(template, cv2.COLOR_RBG2GRAY)
        screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_RBG2GRAY)
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_v >= confidence:
            h, w = template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            self.target_loc = (center_x, center_y)
            return self.target_loc

    def pixel_to_grid_coords(self, px_x: int, px_y: int) -> tuple[int, int]:
        grid_x = (px_x / self.screen_size.width) * self.grid_size[1]
        grid_y = (px_y / self.screen_size.height) * self.grid_size[0]
        grid_x = max(0, min(self.grid_size[1] - 1, grid_x))
        grid_y = max(0, min(self.grid_size[0] - 1, grid_y))
        return (grid_x, grid_y)
