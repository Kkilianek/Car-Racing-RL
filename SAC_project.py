import gymnasium as gym
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.resize_observation import ResizeObservation
from rl_zoo3.wrappers import FrameSkip, HistoryWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(render_mode: str | None = None):
    env = gym.make('CarRacing-v2', render_mode=render_mode)
    env = FrameSkip(env, skip=2)
    env = ResizeObservation(env, shape=64)
    env = GrayScaleObservation(env, keep_dim=True)
    return env


env = make_vec_env(make_env, n_envs=2)
# check if sac_carracing.zip exists, if it does, load it
try:
    model = SAC.load("sac_carracing")
    print("Loaded existing model")
except Exception as e:
    print("Creating new model")

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=7.3e-4,
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.02,
        train_freq=8,
        gradient_steps=10,
        learning_starts=1000,
        use_sde=True,
        use_sde_at_warmup=True,
    )

    # Train model
    model.learn(total_timesteps=int(1e6))
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Save model
    model.save("sac_carracing")

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
