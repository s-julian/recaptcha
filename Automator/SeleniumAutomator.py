from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from Automator.AutomatorInterface import AutomatorInterface
import time

class SeleniumAutomator(AutomatorInterface):
    def __init__(self, url: str, driver_path: str = "chromedriver", headless: bool = False):
        self.url = url
        self.driver_path = driver_path
        self.driver = None
        self.headless = headless
        self.captcha_box = None
        self.reference_element = None
        self.mapper = None

    def launch(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
        else:
            options.add_argument("--start-maximized")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get(self.url)
        print(f"[INFO] Launched browser at {self.url} | Headless: {self.headless}")

    def find_captcha_box(self, by=By.CLASS_NAME, value="g-recaptcha"):
        try:
            iframe = self.driver.find_element(By.XPATH, "//iframe[contains(@src, 'recaptcha')]")
            self.driver.switch_to.frame(iframe)
            box = self.driver.find_element(by, value)
            location = box.location
            size = box.size
            self.captcha_box = {
                "x": location['x'],
                "y": location['y'],
                "width": size['width'],
                "height": size['height']
            }
            print(f"[INFO] CAPTCHA box found at {self.captcha_box}")
            return self.captcha_box
        except NoSuchElementException:
            print(f"[ERROR] CAPTCHA box not found using ({by}, {value}).")
        return None

    def click_box(self):
        try:
            iframe = self.driver.find_element(By.XPATH, "//iframe[contains(@src, 'recaptcha')]")
            self.driver.switch_to.frame(iframe)
            time.sleep(1)

            checkbox = self.driver.find_element(By.ID, "recaptcha-anchor")
            checkbox.click()
            print("[INFO] Clicked reCAPTCHA checkbox.")
            self.driver.switch_to.default_content()

        except NoSuchElementException:
            print("[ERROR] Checkbox not found.")

    def move_mouse_to(self, x, y):
        ActionChains(self.driver).move_to_element_with_offset(
            self.reference_element, x, y
        ).perform()
        print(f"[INFO] Moved to pixel ({x}, {y})")

    def move_to_tile(self, row, col):
        x, y = self.mapper.tile_to_pixel(row, col)
        self.move_mouse_to(x, y)
        print(f"[INFO] Moved to tile ({row}, {col}) → pixel ({x}, {y})")

    def follow_path(self, tile_path):
        for (r, c) in tile_path:
            self.move_to_tile(r, c)
            time.sleep(0.05)

    def refresh(self):
        self.driver.refresh()
        time.sleep(2)

    def close(self):
        self.driver.quit()
        print("[INFO] Closed browser.")
