import platform
import random
import time

import pyautogui

# Configure PyAutoGUI
pyautogui.FAILSAFE = True  # Move mouse to top-left corner to stop script
pyautogui.PAUSE = 0.005  # Small pause between actions

# Detect operating system
IS_MAC = platform.system() == "Darwin"
CMD_KEY = "command" if IS_MAC else "ctrl"  # Used for most shortcuts
CTRL_KEY = "ctrl"  # Used for tab switching on all platforms

# WEBSITE_URL = "https://www.google.com"
WEBSITE_URL = "https://www.google.com/recaptcha/api2/demo"
LOOP_COUNT = 5
WAIT_TIME = 0.005


def navigate_to_website(url):
    """Navigate to a specific website from the
    address bar (assumes it's focused)"""
    # Random mouse movement
    screen_width, screen_height = pyautogui.size()
    random_x = random.randint(100, screen_width - 100)
    random_y = random.randint(100, screen_height - 100)
    pyautogui.moveTo(random_x, random_y, duration=0.1)
    pyautogui.typewrite(url)
    time.sleep(0.1)
    pyautogui.press("enter")


def open_new_tab():
    """Open a new tab using Cmd+T (Mac) or Ctrl+T (Windows)"""
    pyautogui.hotkey(CMD_KEY, "t")
    time.sleep(0.1)


def close_previous_tab():
    """Switch to previous tab and close it"""
    # Use Ctrl+Shift+Tab for tab switching on all platforms
    pyautogui.hotkey(CTRL_KEY, "shift", "tab")
    time.sleep(0.5)
    # Then close tab using Cmd+W (Mac) or Ctrl+W (Windows)
    pyautogui.hotkey(CMD_KEY, "w")
    time.sleep(0.1)


def browser_setup():
    navigate_to_website(WEBSITE_URL)
    idle(1.0)


def browser_teardown():
    open_new_tab()
    close_previous_tab()
    idle(1.0)


def move_mouse(px, py):
    pyautogui.moveTo(px, py, duration=0.005)


def mouse_click(px, py):
    pyautogui.click(px, py)
    idle()


def idle(sleep_time=0.5):
    time.sleep(sleep_time)


if __name__ == "__main__":
    time.sleep(3)
    start = time.time()
    print("Starting browser automation script...")
    print(f"Operating System: {'macOS' if IS_MAC else 'Windows'}")
    print(f"Using {'Command' if IS_MAC else 'Ctrl'} key for shortcuts")
    print("Move mouse to top-left corner to emergency stop")
    print(f"Will loop {LOOP_COUNT} times")
    try:
        for i in range(LOOP_COUNT):
            browser_setup()
            idle()
            browser_teardown()
    except pyautogui.FailSafeException:
        print("\nScript stopped by failsafe (mouse moved to top-left corner)")
    except KeyboardInterrupt:
        print("\nScript stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    print(f"elapsed time: {time.time() - start}")
    print("Script finished.")
