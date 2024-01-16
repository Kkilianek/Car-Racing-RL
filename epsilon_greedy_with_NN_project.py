import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CarRacing-v2", render_mode="human")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        action = action.view(state.shape[0], -1)
        x = torch.cat([state.view(state.shape[0], -1), action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class EpsilonGreedyCarRacingAgent:
    def __init__(self, state_size, action_size, learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float, final_epsilon: float, discount_factor: float):
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.training_errors = []

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, device=device,
                                         dtype=torch.float32).unsqueeze(0)
            action_value = self.q_network(state, action_tensor)
            return action if action_value > 0 else -action

    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               terminated: bool, next_state: np.ndarray) -> None:
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, device=device,
                                  dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, device=device, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, device=device, dtype=torch.float32)

        future_q_value = (not terminated) * self.q_network(next_state, action).detach()
        current_q_value = self.q_network(state, action)
        loss = F.mse_loss(current_q_value,
                          reward + self.discount_factor * future_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_errors.append(loss.item())

    def decay_epsilon(self) -> None:
        """Decays epsilon"""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


learning_rate = 1
n_episodes = 100
initial_epsilon = 1.0
epsilon_decay = initial_epsilon ** (1 / n_episodes)
final_epsilon = 0.01
state_size = np.prod(env.observation_space.shape)
action_size = np.prod(env.action_space.shape)

agent = EpsilonGreedyCarRacingAgent(state_size, action_size, learning_rate,
                                    initial_epsilon, epsilon_decay,
                                    final_epsilon, discount_factor=0.99)
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
