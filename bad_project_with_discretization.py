from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make("CarRacing-v2", render_mode="human")


class EpsilonGreedyCarRacingAgent:
    def __init__(self, learning_rate: float, initial_epsilon: float,
                 epsilon_decay: float, final_epsilon: float, discount_factor: float,
                 n_bins: int):
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.n_bins = n_bins
        self.q_table = defaultdict(
            lambda: np.zeros((n_bins,) * env.action_space.shape[0]))
        self.training_errors = []

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_key = self.state_to_bin(state)
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            action_index = np.unravel_index(np.argmax(self.q_table[state_key]),
                                            self.q_table[state_key].shape)
            return np.array([self.bin_to_action_value(i, bin) for i, bin in
                             enumerate(action_index)])

    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               terminated: bool, next_state: np.ndarray) -> None:
        state_key = self.state_to_bin(state)
        next_state_key = self.state_to_bin(next_state)
        action_key = tuple(self.action_value_to_bin(action))
        future_q_value = (not terminated) * np.max(self.q_table[next_state_key])
        temporal_difference = reward + self.discount_factor * future_q_value - \
                              self.q_table[state_key][action_key]
        self.q_table[state_key][action_key] += self.learning_rate * temporal_difference
        self.training_errors.append(temporal_difference)

    def state_to_bin(self, state: np.ndarray) -> tuple[int, ...]:
        return tuple(
            np.digitize(val, np.linspace(-1, 1, self.n_bins)) for val in state.flatten())

    def action_value_to_bin(self, action_value: np.ndarray) -> tuple[int, ...]:
        return tuple(
            np.digitize(val, np.linspace(low, high, self.n_bins)) for val, low, high in
            zip(action_value, env.action_space.low, env.action_space.high))

    def bin_to_action_value(self, i: int, bin: int) -> float:
        return env.action_space.low[i] + (
                env.action_space.high[i] - env.action_space.low[i]) * (
                bin + 0.5) / self.n_bins

    def decay_epsilon(self) -> None:
        """Decays epsilon"""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


learning_rate = 1
n_episodes = 100
initial_epsilon = 1.0
epsilon_decay = initial_epsilon ** (1 / n_episodes)
final_epsilon = 0.01

agent = EpsilonGreedyCarRacingAgent(learning_rate, initial_epsilon, epsilon_decay,
                                    final_epsilon, discount_factor=0.99, n_bins=100)
env = gym.wrappers.RecordEpisodeStatistics(env)
for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, terminated, next_state)
        state = next_state
        done = terminated or truncated
    agent.decay_epsilon()
