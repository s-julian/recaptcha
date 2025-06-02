import os
import time

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def get_torch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"torch device: {device}")
    return device


def flatten_dict_observation_space(dict_space):
    """
    Flatten a Dict observation space to a Box space for easier processing.
    """
    total_dim = 0
    low_list = []
    high_list = []

    for key in sorted(dict_space.spaces.keys()):
        space = dict_space.spaces[key]
        if isinstance(space, spaces.Box):
            total_dim += np.prod(space.shape)
            low_list.extend(space.low.flatten())
            high_list.extend(space.high.flatten())
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    return spaces.Box(
        low=np.array(low_list, dtype=np.float32),
        high=np.array(high_list, dtype=np.float32),
        dtype=np.float32,
    )


def flatten_dict_observation(obs_dict):
    """
    Flatten a dict observation to a flat array.
    """
    flat_obs = []
    for key in sorted(obs_dict.keys()):
        flat_obs.extend(obs_dict[key].flatten())
    return np.array(flat_obs, dtype=np.float32)


class FlattenDictWrapper(gymnasium.ObservationWrapper):
    """
    Wrapper to flatten Dict observations to Box observations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_dict_observation_space(
            env.observation_space
        )

    def observation(self, obs_dict):
        return flatten_dict_observation(obs_dict)


def make_screengridworld_env(
    target_img_path, matrix_size=1000, render_mode=None
):
    """
    Create single ScreenGridWorldEnv environment for SB3.
    """

    def _init():
        from envs.gridworld_env import ScreenGridWorldEnv

        env = ScreenGridWorldEnv(
            target_img_path=target_img_path,
            matrix_size=matrix_size,
            max_steps=matrix_size * matrix_size,
            render_mode=render_mode,
        )
        # Wrap with flattening wrapper
        env = FlattenDictWrapper(env)
        # Wrap with Monitor for logging
        env = Monitor(env)
        return env

    return _init


class ProgressCallback:
    """Custom callback to print training progress."""

    def __init__(self, print_freq=1000):
        self.print_freq = print_freq
        self.step_count = 0

    def __call__(self, locals_, globals_):
        self.step_count += 1
        if self.step_count % self.print_freq == 0:
            print(f"Training step: {self.step_count}")
        return True


def train_screengridworld_sb3(
    target_img_path,
    matrix_size=1000,
    total_timesteps=500000,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    log_dir="./screengridworld_logs",
    save_freq=25000,
    eval_freq=10000,
    verbose=1,
):
    """
    Train PPO on ScreenGridWorldEnv using Stable-Baselines3.

    Args:
        target_img_path: Path to target image
        matrix_size: Grid world matrix size
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: Maximum value for the gradient clipping
        log_dir: Directory for logs and model checkpoints
        save_freq: Save model every save_freq timesteps
        eval_freq: Evaluate model every eval_freq timesteps
        verbose: Verbosity level

    Returns:
        Trained PPO model
    """

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Set up logger
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    print("Creating ScreenGridWorld environment...")
    print(
        "Note: Using single environment due to pyautogui/browser limitations"
    )
    device = get_torch_device()

    # Create environment
    env_fn = make_screengridworld_env(
        target_img_path=target_img_path,
        matrix_size=matrix_size,
        render_mode=None,
    )

    # Create vectorized environment (single env)
    env = DummyVecEnv([env_fn])

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create PPO model
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=log_dir,
        policy_kwargs={
            "net_arch": [64, 64, 64],  # 3 hidden layers with 64 units each
        },
        device=device,
    )

    # Set custom logger
    model.set_logger(logger)

    print("PPO model created with the following parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training epochs per update: {n_epochs}")
    print("  Policy network: 3 layers, 64 units each")

    # Setup callbacks
    callbacks = []

    # Save model periodically
    if save_freq > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="ppo_screengridworld",
        )
        callbacks.append(checkpoint_callback)

    # Optional: Create evaluation environment and callback
    if eval_freq > 0:
        eval_env = DummyVecEnv([env_fn])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        callbacks.append(eval_callback)

    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Model checkpoints every {save_freq} timesteps")
    if eval_freq > 0:
        print(f"Evaluation every {eval_freq} timesteps")

    start_time = time.time()

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,  # Show progress bar
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return model


def test_trained_model(
    model_path, target_img_path, matrix_size=1000, n_episodes=5
):
    """
    Test a trained model.

    Args:
        model_path: Path to saved model
        target_img_path: Path to target image
        matrix_size: Grid world matrix size
        n_episodes: Number of test episodes
    """

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create test environment
    env_fn = make_screengridworld_env(
        target_img_path=target_img_path,
        matrix_size=matrix_size,
        render_mode=None,
    )
    env = DummyVecEnv([env_fn])

    print(f"Testing model for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            # Print step info
            if (
                episode_length % 10000 == 0 or episode_reward > 0.0
            ):  # Print every 10 steps
                print(f"  Step {episode_length}: Reward = {reward[0]}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1} finished:")
        print(f"  Total reward: {episode_reward}")
        print(f"  Episode length: {episode_length}")
        print(f"  Success: {'Yes' if episode_reward > 0 else 'No'}")

    print("\n--- Test Results ---")
    print(
        f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(
        f"Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%"
    )

    env.close()


if __name__ == "__main__":
    # Training configuration
    target_img_path = "data/target.png"

    print("Starting PPO training with Stable-Baselines3...")
    print("ScreenGridWorld Environment - Single Browser Session")

    # Train the model
    model = train_screengridworld_sb3(
        target_img_path=target_img_path,
        matrix_size=100,
        total_timesteps=500000,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        log_dir="./gridworld_ppo_logs",
        save_freq=25000,  # Save every 25k steps
        eval_freq=0,  # Evaluate every 10k steps
        verbose=1,
    )

    print("\nTraining completed!")

    # Test the trained model
    print("\nTesting the trained model...")
    test_trained_model(
        model_path="./gridworld_ppo_logs/final_model.zip",
        target_img_path=target_img_path,
        matrix_size=100,
        n_episodes=10,
    )

    print("\nAll done!")
