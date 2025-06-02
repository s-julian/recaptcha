import pyautogui
import platform
import subprocess
import time
import random
import pyperclip
import os
from Automator.AutomatorInterface import AutomatorInterface
import Utils.Config as cfg
from GridVisualizer import GridVisualizer

class PyAutoGuiAutomator(AutomatorInterface):
    def __init__(self, browser:str ='chrome', url:str = ""):
        self.browser = browser
        self.os = platform.system()
        self.url = url
        self.last_click_pos = None

    def launch(self, url:str):
        if self.os == "Windows":
            subprocess.Popen(['start', self.browser], shell=True)
        elif self.os == "Darwin":  
            subprocess.Popen(['open', '-a', 'Google Chrome'])
        elif self.os == "Linux":
            subprocess.Popen(['google-chrome'])
        else:
            raise Exception("Unsupported OS")
        
        time.sleep(1)
        if self.os == "Darwin": 
            pyautogui.hotkey('command', 'n')
        else:  
            pyautogui.hotkey('ctrl', 'n')
        
        if (url):
            self.go_to_url(url)
    
    def find_captcha_box(self):
        pass

    def click_box(self):
        self.click_at(cfg.V2_CHECKBOX_POS)
        #self.click_at(cfg.V2_LOCAL_BOX)

    def move_mouse_to(self, x:int, y:int):
        pyautogui.moveTo(x, y, duration=random.uniform(0.3, 0.7))
        time.sleep(random.uniform(0.2, 0.5))
        print(f"[INFO] Moused mouse to ({x}, {y})")

    def move_to_tile(self, row: int, col: int): 
        pass

    def follow_path(self, path: list): 
        pass

    def refresh(self): 
        pass

    def close_browser(self):
        print("[INFO] Attempting to close browser...")
        if self.os == "Windows":
            subprocess.call(["taskkill", "/IM", "chrome.exe", "/F"])
        elif self.os == "Darwin":
            subprocess.call(["pkill", "-f", "Google Chrome"])
        elif self.os == "Linux":
            subprocess.call(["pkill", "-f", "chrome"])
        else:
            print("[WARN] OS not supported for auto browser kill.")
    
    def go_to_url(self, url: str):
        time.sleep(1)
        if self.os == "Darwin":
            pyautogui.hotkey('command', 't')
        else:
            pyautogui.hotkey('ctrl', 't')
        time.sleep(1)
        pyautogui.typewrite(url, interval=0.05)
        pyautogui.press('enter')
    
    def click_at(self, coords: tuple):
        x, y = coords[0], coords[1]
        pyautogui.moveTo(x, y, duration=random.uniform(0.3, 0.7))
        time.sleep(random.uniform(0.2, 0.5))
        pyautogui.click()
        self.last_click_pos = (x, y)
        print(f"[INFO] Clicked at ({x}, {y})")

    
    def copy_token(self):
        coord = cfg.V3_TOKEN_POS
        pos_x, pos_y = coord[0], coord[1]
        print(f"[INFO] Moving to token at ({pos_x}, {pos_y})")
        pyautogui.moveTo(pos_x, pos_y, duration=0.5)
        time.sleep(0.2)
        pyautogui.click(clicks=3, interval=0.1)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.5)
        token = pyperclip.paste()
        print(f"[INFO] Retrieved token: {token[:40]}...")
        return token


    def random_direction(self, steps: int = 100, size: int = 60, delay:float = 0.1):
        for i in range(steps):
            x,y = pyautogui.position()
            direction = random.choice(['u','d','l','r','t'])
            
            if direction == 'u':
                y = max(0, y - size)
            elif direction == 'd':
                y = min(pyautogui.size().height - 1, y + size)
            elif direction == 'l':
                x = max(0, x - size)
            elif direction == 'r':
                x = min(pyautogui.size().width - 1, x + size)
            elif direction == 't':
                if self.os == "Darwin":
                    pyautogui.hotkey('command', 't')
                else:
                    pyautogui.hotkey('ctrl', 't')

            pyautogui.moveTo(x, y, duration=0.1)
            if direction == 't':
                pyautogui.click()
            time.sleep(delay)
    
    def challenge_triggered(self) -> bool:
        try:
            image_path = os.path.abspath("Automator/images/challenge_grid.png")
            location = pyautogui.locateOnScreen(image_path, confidence=0.8)
            if location:
                print("[INFO] Challenge detected on screen.")
                return True
            else:
                print("[INFO] No challenge detected.")
                return False
        except Exception as e:
            print(f"[WARNING] Could not locate challenge image: {e}")
            return False

    def find_mouse_position(self):
        x, y = pyautogui.position()
        print(f"X: {x}, Y: {y}")

    def grid_to_screen(self, x, y, grid_width=20, grid_height=20, origin=(200, 200), scale=40):
        """
        Convert grid coordinates to real screen coordinates.
        - origin: top-left corner of the grid on screen
        - scale: how many pixels one grid step represents
        """
        screen_x = origin[0] + x * scale
        screen_y = origin[1] + y * scale
        return screen_x, screen_y
    
    def test_policy(self, policy_fn, steps: int = 200, delay: float = 0.1):
        """
        Test the given policy function in a live environment.

        Args:
            policy_fn (callable): A function that takes the current position (x, y)
                                and returns a new target position (x', y').
            steps (int): Maximum number of steps to test.
            delay (float): Time to wait between steps.
        """
        print("[INFO] Starting policy test in live browser.")
    
        grid_x, grid_y = 0, 0  # Initial grid position (can randomize if needed)
        visualizer = GridVisualizer()

        for step in range(steps):
            next_grid_x, next_grid_y = policy_fn(grid_x, grid_y)
            screen_x, screen_y = self.grid_to_screen(next_grid_x, next_grid_y)
            self.move_mouse_to(screen_x, screen_y)

            visualizer.update((next_grid_x, next_grid_y))

            if self.challenge_triggered():
                print(f"[FAILURE] Challenge triggered at step {step}.")
                visualizer.close()
                return False

            grid_x, grid_y = next_grid_x, next_grid_y
            time.sleep(delay)

        print("[SUCCESS] Completed all steps without triggering challenge.")
        visualizer.close()
        return True

    def clicked_near_checkbox(self, threshold=40):
        """
        Return True if last click was within `threshold` pixels of the checkbox.
        """
        if self.last_click_pos is None:
            return False

        cx, cy = cfg.V2_CHECKBOX_POS  # your known pixel coordinates
        lx, ly = self.last_click_pos

        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        print(f"[DEBUG] Click distance from checkbox: {dist:.2f}px")

        return dist <= threshold