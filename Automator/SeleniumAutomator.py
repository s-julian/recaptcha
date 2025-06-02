from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from Automator.AutomatorInterface import AutomatorInterface
import time
import Utils.Config as cfg

class SeleniumAutomator(AutomatorInterface):
    def __init__(self, url: str = "", driver_path: str = "chromedriver", headless: bool = False):
        self.url = url
        self.driver_path = driver_path
        self.driver = None
        self.headless = headless
        self.captcha_box = None
        self.reference_element = None
        self.mapper = None

    def launch(self, url: str = ""):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
        else:
            options.add_argument("--start-maximized")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get(url)
        print(f"[INFO] Launched browser at {self.url} | Headless: {self.headless}")
    
    def go_to_url(self, url: str):
        self.driver.get(url)
        time.sleep(0.5)
        print(f"[INFO] Navigated to: {url}")
    
    def click_at(self, element_selector: str):
        try:
            element = self.driver.find_element("css selector", element_selector)
            element.click()
            print(f"[INFO] Clicked Selenium element: {element_selector}")
        except Exception as e:
            print(f"[ERROR] Failed to click element: {e}")

    def find_captcha_box(self, by=By.ID):
        value = cfg.V2_CHECKBOX_SELECTOR
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
        selector = cfg.V2_CHECKBOX_SELECTOR
        try:
            self.find_captcha_box()
            checkbox = self.driver.find_element(By.ID, selector)
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

    def close_browser(self):
        self.driver.quit()
        print("[INFO] Closed browser.")
    
    def copy_token(self, input_field_id="recaptcha-token", timeout=10):
        for _ in range(timeout):
            try:
                token = self.driver.execute_script(
                    f"return document.getElementById('{input_field_id}').value;"
                )
                if token:
                    print(f"[INFO] Retrieved reCAPTCHA token: {token[:30]}...")
                    return token
            except Exception as e:
                print(f"[WARN] Error while fetching token: {e}")
            time.sleep(1)

        print("[ERROR] Token not retrieved within timeout.")
        return None
