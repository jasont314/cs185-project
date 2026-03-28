"""Shared vectorized rollout collection for all PPO agents."""

from typing import Any, Dict

import numpy as np
import torch


def collect_rollouts_vec(
    env: Any,
    policy: torch.nn.Module,
    value_fn: torch.nn.Module,
    buffer: Any,
    num_steps: int,
    device: torch.device,
    agent_state: Dict[str, Any],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, float]:
    """Collect rollouts from a vectorized (or single) environment.

    Works with both gym.Env and gym.vector.VectorEnv. When env is
    vectorized (num_envs > 1), collects num_envs transitions per step,
    giving a ~num_envs× speedup.

    Args:
        env: Gymnasium environment (single or vectorized).
        policy: StochasticFlowPolicy.
        value_fn: ValueFunction.
        buffer: RolloutBuffer.
        num_steps: Steps per environment to collect.
        device: Torch device.
        agent_state: Mutable dict with keys '_last_obs', '_current_ep_rewards',
            '_current_ep_lengths', 'total_steps'. Modified in-place.

    Returns:
        Dict with rollout statistics.
    """
    buffer.reset()
    policy.eval()
    value_fn.eval()

    # Detect vectorized env
    is_vec = hasattr(env, 'num_envs')
    num_envs = env.num_envs if is_vec else 1

    # Initialize or restore state
    if agent_state.get('_last_obs') is None:
        obs, _ = env.reset()
    else:
        obs = agent_state['_last_obs']

    if not is_vec:
        # Ensure obs is 2D for consistency: (1, obs_dim)
        if np.ndim(obs) == 1:
            obs = obs[np.newaxis, :]

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []

    # Per-env episode tracking
    current_ep_rewards = agent_state.get('_current_ep_rewards')
    current_ep_lengths = agent_state.get('_current_ep_lengths')
    if current_ep_rewards is None or len(current_ep_rewards) != num_envs:
        current_ep_rewards = np.zeros(num_envs)
    if current_ep_lengths is None or len(current_ep_lengths) != num_envs:
        current_ep_lengths = np.zeros(num_envs, dtype=int)

    with torch.no_grad():
        for _ in range(num_steps):
            # obs shape: (num_envs, obs_dim)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            # Batched policy forward pass (num_envs samples at once)
            action, info = policy.get_action_and_info(obs_t)
            value = value_fn(obs_t)

            # To numpy — keep batch dim
            action_np = action.cpu().numpy()          # (num_envs, action_dim)
            value_np = value.cpu().numpy()             # (num_envs,)
            hlp_np = info["holistic_log_prob"].cpu().numpy()  # (num_envs,)
            pslp_np = info["per_step_log_probs"].cpu().numpy()  # (num_envs, K)
            # latents: list of (num_envs, latent_dim)
            latents_np = [z.cpu().numpy() for z in info["latents"]]
            noises_np = [e.cpu().numpy() for e in info["noises"]]

            action_clipped = np.clip(action_np, -1.0, 1.0)

            # Step environment
            if is_vec:
                next_obs, rewards, terminated, truncated, infos = env.step(action_clipped)
                dones = np.logical_or(terminated, truncated)
            else:
                act = action_clipped[0]  # single env expects 1D
                next_obs_single, reward, term, trunc, env_info = env.step(act)
                next_obs = next_obs_single[np.newaxis, :]
                rewards = np.array([reward])
                dones = np.array([term or trunc])

            # Store each env's transition
            for e in range(num_envs):
                buffer.add(
                    state=obs[e],
                    action=action_clipped[e],
                    reward=float(rewards[e]),
                    done=float(dones[e]),
                    value=float(value_np[e]),
                    holistic_log_prob=float(hlp_np[e]),
                    per_step_log_probs=pslp_np[e],
                    latents=[z[e] for z in latents_np],
                    noises=[n[e] for n in noises_np],
                )

                current_ep_rewards[e] += rewards[e]
                current_ep_lengths[e] += 1
                agent_state['total_steps'] += 1

                if dones[e]:
                    ep_rewards.append(float(current_ep_rewards[e]))
                    ep_lengths.append(int(current_ep_lengths[e]))
                    current_ep_rewards[e] = 0.0
                    current_ep_lengths[e] = 0

            # Vec envs auto-reset; single env needs manual reset
            if not is_vec and dones[0]:
                next_obs_single, _ = env.reset()
                next_obs = next_obs_single[np.newaxis, :]

            obs = next_obs

    # Save state
    agent_state['_last_obs'] = obs
    agent_state['_current_ep_rewards'] = current_ep_rewards
    agent_state['_current_ep_lengths'] = current_ep_lengths

    # Bootstrap value
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        # For vec env, bootstrap per-env; for single, take first
        last_values = value_fn(obs_t).cpu().numpy()
        last_value = float(last_values[0]) if num_envs == 1 else float(last_values.mean())

    buffer.compute_returns(last_value, gamma=gamma, gae_lambda=gae_lambda)

    return {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "num_episodes": len(ep_rewards),
    }
