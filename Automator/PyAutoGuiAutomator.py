import pyautogui
import platform
import subprocess
import time
import random
import pyperclip
from Automator.AutomatorInterface import AutomatorInterface
import Utils.Config as cfg

class PyAutoGuiAutomator(AutomatorInterface):
    def __init__(self, browser:str ='chrome', url:str = ""):
        self.browser = browser
        self.os = platform.system()
        self.url = url

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
    
    def find_captcha_box(self):
        pass

    def click_box(self):
        #self.click_at(cfg.V2_CHECKBOX_POS)
        self.click_at(cfg.V2_LOCAL_BOX)

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
        location = pyautogui.locateOnScreen('images/challenge_grid.png', confidence=0.8)
        if location:
            print("[INFO] Challenge detected on screen.")
            return True
        else:
            print("[INFO] No challenge detected.")
            return False

    def find_mouse_position(self):
        x, y = pyautogui.position()
        print(f"X: {x}, Y: {y}")