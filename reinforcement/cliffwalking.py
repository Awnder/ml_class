from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

class CliffWalkingAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a reinforcement learning agent for the Cliff Walking environment.
        Args:
            env (gym.Env): The Cliff Walking environment.
            learning_rate (float): The learning rate for the agent.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decreases.
            final_epsilon (float): The minimum exploration rate.
            discount_factor (float): Discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Returns best action with probability (1 - epsilon).
        Otherwise returns random action with probability epsilon.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool]
    ) -> None:
        """Updates the Q-Value of the action
        Args:
            obs (tuple[int, int, bool]): The current observation.
            action (int): The action taken.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            next_obs (tuple[int, int, bool]): The next observation after taking the action.
        """
        future_q = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference_error = (
            reward + self.discount_factor * future_q - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference_error
        )

        self.training_error.append(temporal_difference_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    # hyperparameters
    learning_rate = 0.01
<<<<<<< HEAD
    n_episodes = 1000
=======
    n_episodes = 100_000
>>>>>>> 4cf352b39261af82af4ca9d8620fc1c57be12cf9
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    env = gym.make("CliffWalking-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = CliffWalkingAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

