"""Evaluation utilities for stochastic flow policies."""

from typing import Any, Dict, Optional

import numpy as np
import torch

from src.models.stochastic_flow_policy import StochasticFlowPolicy


def evaluate_policy(
    env: Any,
    policy: StochasticFlowPolicy,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a stochastic flow policy over multiple episodes.

    For deterministic evaluation, sigma is temporarily set to zero so the
    denoising chain is noise-free (pure drift). This is restored after
    evaluation.

    Args:
        env: A gymnasium-compatible environment.
        policy: The stochastic flow policy to evaluate.
        num_episodes: Number of evaluation episodes to run.
        deterministic: If True, disable stochastic noise (set sigma to 0).
        device: Torch device. Inferred from policy parameters if not given.

    Returns:
        Dict with keys:
            mean_return: Mean episodic return across episodes.
            std_return: Std of episodic returns.
            returns_list: List of per-episode returns.
    """
    if device is None:
        device = next(policy.parameters()).device

    policy.eval()

    # For deterministic evaluation, temporarily zero out sigma
    saved_sigma_state = None
    if deterministic:
        saved_sigma_state = _set_sigma_zero(policy)

    returns_list: list[float] = []

    try:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0

            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                )
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)

                with torch.no_grad():
                    action = policy(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                episode_return += float(reward)

            returns_list.append(episode_return)
    finally:
        # Restore sigma even if an error occurs
        if deterministic and saved_sigma_state is not None:
            _restore_sigma(policy, saved_sigma_state)

    returns_arr = np.array(returns_list)
    return {
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "returns_list": returns_list,
    }


def _set_sigma_zero(policy: StochasticFlowPolicy) -> dict:
    """Set sigma to zero for deterministic evaluation.

    Returns a state dict that can be used to restore the original sigma.
    """
    saved = {}

    if policy.sigma_network_flag:
        # For sigma networks, save state dict and replace with zero-output
        saved["type"] = "network"
        saved["state_dict"] = {
            k: v.clone() for k, v in policy.sigma_net.state_dict().items()
        }
        # Zero out all parameters so softplus outputs ~0
        with torch.no_grad():
            for param in policy.sigma_net.parameters():
                param.fill_(-100.0)  # softplus(-100) ~ 0
    else:
        saved["type"] = "scalar"
        saved["log_sigma_value"] = policy.log_sigma.clone()
        # Set log_sigma to very negative value so exp(log_sigma) ~ 0
        with torch.no_grad():
            if isinstance(policy.log_sigma, torch.nn.Parameter):
                policy.log_sigma.fill_(-100.0)
            else:
                policy.log_sigma.fill_(-100.0)

    return saved


def _restore_sigma(policy: StochasticFlowPolicy, saved: dict) -> None:
    """Restore sigma from saved state."""
    if saved["type"] == "network":
        policy.sigma_net.load_state_dict(saved["state_dict"])
    else:
        with torch.no_grad():
            if isinstance(policy.log_sigma, torch.nn.Parameter):
                policy.log_sigma.copy_(saved["log_sigma_value"])
            else:
                policy.log_sigma.copy_(saved["log_sigma_value"])
