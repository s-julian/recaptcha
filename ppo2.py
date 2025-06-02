import torch
import torch.nn as nn
from torch.distributions import Categorical


def get_torch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"torch device: {device}")
    return device


device = get_torch_device()

# Implementation derived from https://github.com/nikhilbarhate99/PPO-PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas
        )
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions = self.policy_old.act(state, memory)
        return actions.item()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(
            torch.stack(memory.states).to(device), 1
        ).detach()
        old_actions = torch.squeeze(
            torch.stack(memory.actions).to(device), 1
        ).detach()
        old_logprobs = (
            torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        )

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


if __name__ == "__main__":
    import os

    from envs.grid2 import DiscreteGridWorldEnv

    # Example usage:
    target_image_path = os.path.join(os.getcwd(), "data", "target.png")

    # For training (no real pyautogui actions during steps)
    print("\n--- Training Mode Example ---")
    train_env = DiscreteGridWorldEnv(
        target_img_path=target_image_path,
        matrix_size=100,
        max_steps=10e7,
        render_mode=None,
        is_eval_mode=False,
    )
    obs, info = train_env.reset()
    for _ in range(10000):  # Run for 50 steps
        action = train_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = train_env.step(action)
        if terminated or truncated:
            print("Episode finished in training mode.")
            break
    train_env.close()

    # For evaluation (pyautogui actions on termination)
    print("\n--- Evaluation Mode Example ---")
    eval_env = DiscreteGridWorldEnv(
        target_img_path=target_image_path,
        matrix_size=100,
        max_steps=10e7,
        render_mode="console",
        is_eval_mode=True,
    )
    obs, info = eval_env.reset()
    for _ in range(2000):  # Run for 50 steps
        action = eval_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            print("Episode finished in evaluation mode.")
            break
    eval_env.close()
