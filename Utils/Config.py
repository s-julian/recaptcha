from dotenv import load_dotenv
import os
import json

load_dotenv()

RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")
ANTCPT_URL = os.getenv("ANTCPT_URL")
DEMO_URL = os.getenv("DEMO_URL")
GOOGLE_RECAPTCHA_V2_URL = os.getenv("GOOGLE_RECAPTCHA_V2_URL")
RECAPTCHA_SITE_VERIFY_URL = os.getenv("RECAPTCHA_SITE_VERIFY_URL")

with open("config.json") as f:
    data = json.load(f)

V2_CHECKBOX_POS = tuple(data["v2_checkbox_position"])
V3_TOKEN_POS = tuple(data["v3_token_position"])
V2_CHECKBOX_SELECTOR = data["v2_checkbox_selector"]
V2_LOCAL_BOX = data["local_pos"]