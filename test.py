import torch
from Automator.SeleniumAutomator import SeleniumAutomator
import time
from selenium.webdriver.common.by import By

def main():
    pass
    '''if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"torch device: {device}")'''


if __name__ == "__main__":
    automator = SeleniumAutomator(url="https://www.google.com/recaptcha/api2/demo")
    automator.launch()
    time.sleep(5)
    automator.find_captcha_box(By.ID, "recaptcha-anchor")
    #automator.find_captcha_box(By.CLASS_NAME, "g-recaptcha")
    #automator.find_captcha_box(By.CSS_SELECTOR, "#captcha-box-id")
    automator.click_box()
    time.sleep(5)
    automator.close()
