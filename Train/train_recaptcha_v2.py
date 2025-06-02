from stable_baselines3 import PPO
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Env.GridWorldMouseEnv import GridWorldMouseEnv
from Env.realEvalWrapper import RealEvalWrapper
from stable_baselines3.common.env_checker import check_env

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
    next_x = int(np.clip(grid_x + dx, 0, 19))
    next_y = int(np.clip(grid_y + dy, 0, 19))

    return next_x, next_y


sim_env = GridWorldMouseEnv()
env = RealEvalWrapper(sim_env, test_interval=10, fn=policy_fn)

check_env(sim_env, warn=True)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_tensorboard/"
)

# --------- Train Agent ---------
for step in range(0, 100_000, 10_000):
    model.learn(total_timesteps=10_000)
    model.save(f"checkpoints/agent_{step + 10_000}")
