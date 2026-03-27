"""Utility functions for probability distributions."""

import torch
import torch.nn.functional as F
from typing import Tuple
import math


def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Compute log probability under a diagonal Gaussian.

    Args:
        mean: Mean of the Gaussian, shape (..., D).
        std: Standard deviation (positive), shape (..., D).
        sample: Sample point, shape (..., D).

    Returns:
        Log probability, shape (...) — summed over the last dimension.
    """
    var = std ** 2
    log_std = torch.log(std)
    log_prob = -0.5 * (
        ((sample - mean) ** 2) / var
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    # Sum over the feature dimension
    return log_prob.sum(dim=-1)


def tanh_squash_correction(pre_tanh_value: torch.Tensor) -> torch.Tensor:
    """Compute the log-det-Jacobian correction for tanh squashing.

    For a = tanh(z), the correction to add to the log-prob of z is:
        -sum_i log(1 - tanh(z_i)^2)

    which equals  -sum_i 2 * (log(2) - z_i - softplus(-2*z_i))
    using the numerically stable form.

    Args:
        pre_tanh_value: The value before tanh, shape (..., D).

    Returns:
        Log-det-Jacobian correction, shape (...) — summed over the last dim.
        This value should be *subtracted* from the log-prob to get the
        log-prob of the squashed variable.
    """
    # Numerically stable: log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
    correction = 2.0 * (math.log(2.0) - pre_tanh_value - F.softplus(-2.0 * pre_tanh_value))
    return correction.sum(dim=-1)


def sample_gaussian(mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from a diagonal Gaussian via the reparameterization trick.

    Args:
        mean: Mean of the Gaussian, shape (..., D).
        std: Standard deviation, shape (..., D).

    Returns:
        sample: The sampled value, shape (..., D).
        eps: The noise used (so sample = mean + std * eps), shape (..., D).
    """
    eps = torch.randn_like(mean)
    sample = mean + std * eps
    return sample, eps
