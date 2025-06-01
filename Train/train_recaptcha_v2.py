from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Env.v2Env import RecaptchaV2Env
from Env.v2Env import RecaptchaV2Env
from stable_baselines3.common.env_checker import check_env


env = RecaptchaV2Env()
check_env(env, warn=True)


#model = PPO("MlpPolicy", env, verbose=1) #PPO("MultiInputPolicy", env, verbose=1)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=5,
    batch_size=2,
    n_epochs=1,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./ppo_logs/"
)
#model.learn(total_timesteps=10000)
model.learn(total_timesteps=5)

model.save("ppo_recaptcha_v2")
print("[INFO] Training complete. Model saved to ppo_recaptcha_v2.zip.")
