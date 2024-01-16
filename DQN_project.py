import random
from collections import namedtuple, deque
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display
from torch import nn
from torch.nn import functional as F

env = gym.make("CarRacing-v2", render_mode="human")

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    pass

plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


BATCH_SIZE = 128
# Gamma is the discount factor
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.shape[0]
state, info = env.reset()

n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state: np.ndarray):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device,
                            dtype=torch.float32)

episode_durations = []

def plot_durations(show_result: bool = False) -> None:
    plt.figure(1)
    duration_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model() -> None:
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(
        1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

n_episodes = 1000
for i_episode in range(n_episodes):
    state, info = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, device=device,
                                      dtype=torch.float32).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        loss = optimize_model()
        if loss is not None:
            print(f"Episode: {i_episode}, Loss: {loss}")

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + \
                                         (1 - TAU) * target_net_state_dict[key]

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    if i_episode % 10 == 0:
        plot_durations(show_result=True)
        print(f"Episode: {i_episode}, Loss: {loss}")
        print(f"Episode: {i_episode}, Duration: {t + 1}")
        print(f"Episode: {i_episode}, Mean Duration: {np.mean(episode_durations[-100:])}")

env.close()


