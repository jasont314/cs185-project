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


def make_vec_env(env_name: str, num_envs: int, seed: int = 0) -> gym.vector.VectorEnv:
    """Create vectorized environments for parallel rollout collection.

    Args:
        env_name: Gymnasium environment ID.
        num_envs: Number of parallel environments.
        seed: Base random seed (each env gets seed + i).

    Returns:
        A vectorized environment.
    """
    def make_single(i: int):
        def _init():
            env = gym.make(env_name)
            env.reset(seed=seed + i)
            return env
        return _init

    if num_envs == 1:
        return gym.vector.SyncVectorEnv([make_single(0)])
    # AsyncVectorEnv runs each env in a separate process,
    # utilizing multiple CPU cores. Falls back to Sync if it fails.
    try:
        return gym.vector.AsyncVectorEnv([make_single(i) for i in range(num_envs)])
    except Exception:
        return gym.vector.SyncVectorEnv([make_single(i) for i in range(num_envs)])


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
