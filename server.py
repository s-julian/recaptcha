import os

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

RECAPTCHA_SECRET = os.getenv("RECAPTCHA_SECRET", "your-secret-key")

app.mount("/static", StaticFiles(directory="static"), name="static")


class RecaptchaToken(BaseModel):
    token: str


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/score")
async def verify_recaptcha(data: RecaptchaToken):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": RECAPTCHA_SECRET, "response": data.token},
        )
        result = response.json()
        return {
            "success": result.get("success", False),
            "score": result.get("score", None),
            "action": result.get("action", None),
        }


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("DEV_SERVER_HOST"),
        port=int(os.getenv("DEV_SERVER_PORT")),
        reload=True,
    )
