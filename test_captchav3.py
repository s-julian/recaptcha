from Automator.PyAutoGuiAutomator import PyAutoGuiAutomator
from Utils.ReCaptchav3 import verify_recaptcha_v3
import Utils.Config as cfg
import time
import argparse

def get_automator_class(automator_type):
    if automator_type == "pyautogui":
        from Automator.PyAutoGuiAutomator import PyAutoGuiAutomator
        return PyAutoGuiAutomator
    elif automator_type == "selenium":
        from Automator.SeleniumAutomator import SeleniumAutomator
        return SeleniumAutomator
    else:
        raise ValueError(f"Unsupported automator: {automator_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reCAPTCHA automation.")
    parser.add_argument(
        "--automator",
        type=str,
        default="pyautogui",
        choices=["pyautogui", "selenium"],
        help="Choose the automation method to use."
    )
    args = parser.parse_args()

    AutomatorClass = get_automator_class(args.automator)
    automator = AutomatorClass()

    try:
        automator.launch(cfg.DEMO_URL)
        automator.go_to_url(cfg.DEMO_URL)
        time.sleep(3)
        token = automator.copy_token()
        verify_recaptcha_v3(token)
    finally:
        automator.close_browser()
