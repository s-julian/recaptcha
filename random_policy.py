import time

from envs.gridworld_env import ScreenGridWorldEnv

# Assuming ScreenGridWorldEnv, GridWorld, Navigator, and pag are correctly imported or defined
# For this example, let's assume they are in the same file or accessible.
# from your_module import ScreenGridWorldEnv, GridWorld, Navigator, pag

# --- (Your ScreenGridWorldEnv, GridWorld, Navigator, and pag definitions would go here if not imported) ---

# Example usage:
if __name__ == "__main__":
    TARGET_IMG = "/Users/julian/Documents/GitHub/school/cs486/recaptcha/data/target.png"

    try:
        # Create the environment
        env = ScreenGridWorldEnv(
            target_img_path=TARGET_IMG, matrix_size=10, render_mode="console"
        )

        num_episodes = 10
        print(
            f"\n--- Running {num_episodes} test episodes with a random policy ---"
        )
        time.sleep(3)
        for i_episode in range(num_episodes):
            print(f"\n===== Episode {i_episode + 1} =====")
            observation, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            step_count = 0

            while not terminated and not truncated:
                # Agent chooses a random action
                action = (
                    env.action_space.sample()
                )  # Pick a random action (0-4)

                print(f"\n--- Step {step_count + 1} ---")
                print(
                    f"Agent Action: {action} ({env._gym_action_to_navigator_direction[action]})"
                )

                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                episode_reward += reward
                step_count += 1

                env.render()  # Render after each step to see progress

            print(
                f"Episode {i_episode + 1} finished after {step_count} steps."
            )
            print(f"Final Reward: {episode_reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        if "env" in locals() and env is not None:
            env.close()
            print("Environment closed.")
