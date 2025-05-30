from Automator.PyAutoGuiAutomator import PyAutoGuiAutomator
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
        automator.launch(cfg.GOOGLE_RECAPTCHA_V2_URL)
        automator.go_to_url(cfg.GOOGLE_RECAPTCHA_V2_URL)
        automator.click_box()
        time.sleep(3)
    finally:
        automator.close_browser()
    