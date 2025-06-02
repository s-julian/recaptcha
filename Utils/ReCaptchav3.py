import requests
import Utils.Config as cfg

def verify_recaptcha_v3(token: str) -> dict:
    payload = {
        'secret': cfg.RECAPTCHA_SECRET_KEY,
        'response': token
    }

    response = requests.post(cfg.RECAPTCHA_SITE_VERIFY_URL, data=payload)
    result = response.json()

    if result.get("success"):
        score = result.get("score", 0.0)
        action = result.get("action", "")
        print(f"[INFO] reCAPTCHA verification passed | Score: {score}, Action: {action}")
    else:
        print("[ERROR] reCAPTCHA verification failed:", result)

    return result