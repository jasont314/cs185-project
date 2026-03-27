"""General math utilities for RL training."""

import numpy as np
import torch
from typing import Union


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the explained variance between predictions and targets.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Returns -1 if y_true has zero variance, 0 if prediction is no better
    than the mean, and 1 for perfect prediction.

    Args:
        y_pred: Predicted values, shape (N,).
        y_true: True values, shape (N,).

    Returns:
        Explained variance as a float.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    var_true = np.var(y_true)
    if var_true == 0.0:
        return -1.0
    return float(1.0 - np.var(y_true - y_pred) / var_true)


def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative sums (generalized returns).

    Given rewards [r_0, r_1, ..., r_{T-1}] and discount gamma, returns:
        [r_0 + gamma*r_1 + gamma^2*r_2 + ...,
         r_1 + gamma*r_2 + ...,
         ...,
         r_{T-1}]

    Uses reverse accumulation for O(T) computation.

    Args:
        rewards: 1-D array of rewards, shape (T,).
        gamma: Discount factor in [0, 1].

    Returns:
        Discounted cumulative sums, shape (T,).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    result = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        result[t] = running
    return result


def normalize(
    x: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize to zero mean and unit variance.

    Args:
        x: Input tensor or array.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized tensor or array with the same type as input.
    """
    if isinstance(x, torch.Tensor):
        return (x - x.mean()) / (x.std() + eps)
    else:
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std() + eps)
