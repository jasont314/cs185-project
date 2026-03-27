"""Environment factory utilities."""

import gymnasium as gym


def make_env(env_name: str, seed: int = 0) -> gym.Env:
    """Create a gymnasium environment.

    Args:
        env_name: Gymnasium environment ID (e.g. "HalfCheetah-v5").
        seed: Random seed for the environment.

    Returns:
        A gymnasium environment instance, already seeded.
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def get_env_info(env_name: str) -> dict:
    """Get state_dim and action_dim for an environment.

    Args:
        env_name: Gymnasium environment ID.

    Returns:
        Dict with keys: state_dim, action_dim, action_high.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    env.close()
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_high": float(action_high),
    }
