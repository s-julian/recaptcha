import pyautogui
import platform
import subprocess
import time
import random

def open():
    if platform.system() == "Windows":
        subprocess.Popen(['start', 'chrome'], shell=True)
    elif platform.system() == "Darwin":  
        subprocess.Popen(['open', '-a', 'Google Chrome'])
    elif platform.system() == "Linux":
        subprocess.Popen(['google-chrome'])
    else:
        raise Exception("Unsupported OS")
    time.sleep(1)
    if platform.system() == "Darwin": 
        pyautogui.hotkey('command', 'n')
    else:  
        pyautogui.hotkey('ctrl', 'n')

def go_to_url(url):
    time.sleep(1)
    pyautogui.hotkey('command', 't')
    time.sleep(1)
    pyautogui.typewrite(url, interval=0.05)
    pyautogui.press('enter')

def random_direction(steps = 100,size = 60,delay = 0.1):
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
           pyautogui.hotkey('command', 't')

        pyautogui.moveTo(x, y, duration=0.1)
        if direction == 't':
            pyautogui.click()
        time.sleep(delay)

#switching tabs (at random moments)
if __name__ == "__main__":
    open()
    go_to_url("https://www.google.com")
    #go_to_url("http://127.0.0.1:8000/")
    random_direction()