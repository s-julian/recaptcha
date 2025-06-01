import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from matplotlib.colors import ListedColormap


class GridWorld:
    def __init__(self, matrix_size: int = 1000):
        self.matrix_size = matrix_size
        self.screenshot = None
        self.grid_matrix = None
        self.mouse_loc = None
        self.target_loc = None
        self.target_dim = None
        self.screenshot_dim = None

    def __take_screenshot_and_mouse_position(self):
        mouse_pos = pyautogui.position()
        screenshot = pyautogui.screenshot()
        self.screenshot = np.array(screenshot)
        screenshot_width, screenshot_height = screenshot.size
        screen_width, screen_height = pyautogui.size()
        self.screenshot_dim = (screenshot_width, screenshot_height)

        # Handle display scaling
        if (screenshot_width, screenshot_height) != (
            screen_width,
            screen_height,
        ):
            scale_x = screenshot_width / screen_width
            scale_y = screenshot_height / screen_height
            mouse_scaled = (
                int(mouse_pos[0] * scale_x),
                int(mouse_pos[1] * scale_y),
            )
        else:
            mouse_scaled = (mouse_pos[0], mouse_pos[1])
        self.mouse_loc = mouse_scaled

    def __perform_template_matching(self, target_img_path: str):
        img_gray = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        assert template is not None, (
            f"Error! Could not read template image: {target_img_path}"
        )
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        self.target_loc = min_loc
        self.target_dim = (w, h)

    def __create_gridworld_matrix(self):
        screenshot_width, screenshot_height = self.screenshot_dim
        target_width, target_height = self.target_dim
        target_x, target_y = self.target_loc
        mouse_x, mouse_y = self.mouse_loc
        side_len = max(screenshot_width, screenshot_height)
        scale_factor = self.matrix_size / side_len

        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)

        # Fill navigable area
        display_w = int(screenshot_width * scale_factor)
        display_h = int(screenshot_height * scale_factor)
        matrix[:display_h, :display_w] = 1

        # Fill target area
        target_x_scaled = int(target_x * scale_factor)
        target_y_scaled = int(target_y * scale_factor)
        target_w_scaled = int(target_width * scale_factor)
        target_h_scaled = int(target_height * scale_factor)
        x_end = min(target_x_scaled + target_w_scaled, self.matrix_size)
        y_end = min(target_y_scaled + target_h_scaled, self.matrix_size)
        matrix[target_y_scaled:y_end, target_x_scaled:x_end] = 2

        # Fill mouse position
        mouse_x_scaled = int(mouse_x * scale_factor)
        mouse_y_scaled = int(mouse_y * scale_factor)
        if (
            0 <= mouse_x_scaled < self.matrix_size
            and 0 <= mouse_y_scaled < self.matrix_size
        ):
            matrix[mouse_y_scaled][mouse_x_scaled] = 3

        self.grid_matrix = matrix

    def __display_gridworld_matrix(self):
        screenshot_width, screenshot_height = self.screenshot_dim
        target_width, target_height = self.target_dim
        target_x, target_y = self.target_loc
        mouse_x, mouse_y = self.mouse_loc
        side_len = max(screenshot_width, screenshot_height)
        scale_factor = self.matrix_size / side_len

        if self.matrix_size >= 1000:
            grid_spacing = 20
            target_min = 40
            mouse_min = 20
        elif self.matrix_size >= 100:
            grid_spacing = 2
            target_min = 4
            mouse_min = 2
        else:
            grid_spacing = 1
            target_min = 2
            mouse_min = 1

        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)

        # Fill navigable
        display_w = int(screenshot_width * scale_factor)
        display_h = int(screenshot_height * scale_factor)
        matrix[:display_h, :display_w] = 1

        # Target
        target_x_scaled = int(target_x * scale_factor)
        target_y_scaled = int(target_y * scale_factor)
        target_w_scaled = max(int(target_width * scale_factor), target_min)
        target_h_scaled = max(int(target_height * scale_factor), target_min)
        x_end = min(target_x_scaled + target_w_scaled, self.matrix_size)
        y_end = min(target_y_scaled + target_h_scaled, self.matrix_size)
        matrix[target_y_scaled:y_end, target_x_scaled:x_end] = 2

        # Mouse
        mouse_x_scaled = int(mouse_x * scale_factor)
        mouse_y_scaled = int(mouse_y * scale_factor)
        half_mouse = mouse_min // 2
        y_start = max(0, mouse_y_scaled - half_mouse)
        y_end = min(self.matrix_size, mouse_y_scaled + half_mouse + 1)
        x_start = max(0, mouse_x_scaled - half_mouse)
        x_end = min(self.matrix_size, mouse_x_scaled + half_mouse + 1)
        matrix[y_start:y_end, x_start:x_end] = 3

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = ListedColormap(["black", "white", "red", "green"])
        ax.imshow(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=3,
            origin="upper",
            interpolation="none",
        )

        for i in range(0, self.matrix_size, grid_spacing):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        ax.set_title(f"Gridworld (n={self.matrix_size})")
        ax.legend(
            handles=[
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor="black", label="Non-navigable"
                ),
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="black",
                    label="Navigable",
                ),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", label="Target"),
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor="green", label="Current Location"
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        path = os.path.join("data", "test", f"gridworld_plot_{timestamp}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # plt.savefig(path)
        plt.show()
        # print(f">> Gridworld image saved to {path}")

    def __save_grid_to_csv(self, output_dir="data/test"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        filename = f"gridworld_matrix_{timestamp}.csv"
        path = os.path.join(output_dir, filename)
        np.savetxt(path, self.grid_matrix, delimiter=",", fmt="%d")
        print(f">> Grid matrix saved to {path}")

    def display_template_matching(self):
        assert self.screenshot is not None, "Screenshot not captured."
        assert self.mouse_loc is not None, "Mouse location not set."
        assert self.target_loc is not None and self.target_dim is not None, (
            "Target location/dim not set."
        )

        img_rgb = cv2.cvtColor(self.screenshot.copy(), cv2.COLOR_RGB2BGR)
        annotated_img = img_rgb.copy()

        # Draw full-screen square boundary
        width, height = self.screenshot_dim  # (width, height)
        side_len = max(width, height)
        cv2.rectangle(
            annotated_img, (0, 0), (side_len, side_len), (0, 0, 0), 10
        )

        # Draw target rectangle
        top_left = self.target_loc
        w, h = self.target_dim
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(annotated_img, top_left, bottom_right, (0, 0, 255), 10)

        # Draw mouse location
        cv2.circle(
            annotated_img,
            self.mouse_loc,
            radius=15,
            color=(0, 255, 0),
            thickness=5,
        )

        # Display and save
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        ax.set_title("Template Match Debug View")
        ax.axis("off")

        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        filename = f"template_match_debug_{timestamp}.png"
        debug_path = os.path.join("data", "test", filename)
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        # plt.savefig(debug_path)
        plt.show()

        print(f">> Debug image saved to {debug_path}")
        print(f">> Target region: top_left={top_left}, size={w}x{h}")
        print(f">> Mouse location: {self.mouse_loc}")

    def build(self, target_img_path: str, save_csv=True):
        self.__take_screenshot_and_mouse_position()
        self.__perform_template_matching(target_img_path)
        self.__create_gridworld_matrix()
        # self.__display_gridworld_matrix()
        if save_csv:
            self.__save_grid_to_csv()


if __name__ == "__main__":
    target_img_path = os.path.join(os.getcwd(), "data", "target.png")
    time.sleep(3)
    gw = GridWorld(matrix_size=1000)
    gw.build(target_img_path=target_img_path, save_csv=False)
    print(
        f"matrix_size={gw.matrix_size}\n"
        f"mouse_loc={gw.mouse_loc}\n"
        f"target_loc={gw.target_loc}\n"
        f"target_dim={gw.target_dim}\n"
        f"screenshot_dim={gw.screenshot_dim}"
    )
