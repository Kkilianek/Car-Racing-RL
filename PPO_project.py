import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.resize_observation import ResizeObservation
from rl_zoo3.wrappers import FrameSkip
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(render_mode: str | None = None):
    env = gym.make('CarRacing-v2', render_mode=render_mode)
    env = FrameSkip(env, skip=2)
    env = ResizeObservation(env, shape=64)
    env = GrayScaleObservation(env, keep_dim=True)
    return env


env = make_vec_env(make_env, n_envs=8)
# check if ppo_carracing.zip exists, if it does, load it
try:
    model = PPO.load("ppo_carracing")
    print("Loaded existing model")
except Exception as e:
    print("Creating new model")
    # Define policy kwargs
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.GELU,
        net_arch=dict(pi=[256], vf=[256]),
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Create PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
    )

    # Train model
    model.learn(total_timesteps=int(4e5))
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Save model
    model.save("ppo_carracing")

# Run 100 episodes
single_env = make_env(render_mode='human')

# Run 100 episodes
for episode in range(100):
    obs, info = single_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = single_env.step(action)
        done = terminated or truncated

# Close environment
single_env.close()