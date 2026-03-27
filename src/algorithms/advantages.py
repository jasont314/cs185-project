"""Generalized Advantage Estimation (GAE) for PPO."""

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns.

    Args:
        rewards: Rewards, shape (T,) or (T, num_envs).
        values: Value predictions including bootstrap, shape (T+1,) or (T+1, num_envs).
        dones: Done flags (1.0 = episode ended), shape (T,) or (T, num_envs).
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: GAE advantages, shape (T,) or (T, num_envs).
        returns: advantages + values[:-1], shape (T,) or (T, num_envs).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)

    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(rewards[0])

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:-1]
    return advantages, returns
