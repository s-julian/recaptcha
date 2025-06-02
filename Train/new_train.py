from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Env.GridWorldMouseEnv import GridWorldMouseEnv
from Env.realEvalWrapper import RealEvalWrapper

# Load model if continuing training, or initialize fresh
model_path = "recaptcha_stealth_agent_2000.zip"
try:
    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")
except:
    model = None
    print("[INFO] Starting fresh training")



# Define policy function using the current model
def policy_fn(grid_x, grid_y):
    obs = np.array([[grid_x / 19, grid_y / 19]], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    moves = {
        0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0),
        4: (-1, -1), 5: (1, -1), 6: (-1, 1), 7: (1, 1),
        8: (0, 0)  # click
    }

    dx, dy = moves.get(action, (0, 0))
    next_x = np.clip(grid_x + dx, 0, 19)
    next_y = np.clip(grid_y + dy, 0, 19)
    return next_x, next_y

# Create and wrap environment
env = GridWorldMouseEnv()
env = RealEvalWrapper(env, test_interval=5, fn=policy_fn)
vec_env = DummyVecEnv([lambda: env])

# Instantiate or reload model
if model is None:
    model = PPO("MlpPolicy", vec_env, verbose=1)

# Train model
total_timesteps = 2000
timesteps_per_eval = 1000

for step in range(0, total_timesteps, timesteps_per_eval):
    model.learn(total_timesteps=timesteps_per_eval)
    model.save(f"recaptcha_stealth_agent_{step + timesteps_per_eval}")
    print(f"[INFO] Model saved at step {step + timesteps_per_eval}")

print("✅ Training complete.")
