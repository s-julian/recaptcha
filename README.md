# reCAPTCHAv3 RL Agent
## CS 486 Section 1 Project

### Setup Project
Install project requirements with the following command: ```pip install -r requirements.txt```


### Run Project
Test pytorch environment by running test.py with command ```python3 test.py```. Output should specify what device is being used by pytorch (CUDA, Apple MPS, or CPU)

Test FastAPI by running server.py with command ```fastapi dev server.py```.
- visit the following endpoints to verify the server is running:
	- http://127.0.0.1:8000/
	- http://127.0.0.1:8000/items/5?q=somequery
	- http://127.0.0.1:8000/redoc

### Documentation
- Gymnasium: https://gymnasium.farama.org/introduction/basic_usage/
- Selenium: https://www.selenium.dev/documentation/
- PyAutoGUI: https://pyautogui.readthedocs.io/en/latest/index.html
- FastAPI: https://fastapi.tiangolo.com
- Polars: https://docs.pola.rs