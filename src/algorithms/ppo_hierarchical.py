"""PPO with hierarchical advantage: holistic baseline + learned corrections.

A_{t,k} = A_t / K + delta_k(s, z_k, k)

where delta_k sums to 0 by construction. This preserves the holistic
advantage as a guaranteed floor while learning non-uniform credit
redistribution across internal flow steps.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.buffers import RolloutBuffer
from src.algorithms.losses import (
    compute_entropy_bonus,
    compute_hierarchical_ppo_loss,
    compute_value_loss,
)
from src.models.stochastic_flow_policy import StochasticFlowPolicy
from src.models.value_function import ValueFunction
from src.models.weighting_network import HierarchicalCorrectionNetwork


_DEFAULT_CONFIG: Dict[str, Any] = {
    "state_dim": None,
    "action_dim": None,
    "hidden_dim": 256,
    "latent_dim": None,
    "K": 4,
    "sigma_init": 1.0,
    "learn_sigma": False,
    "sigma_network": False,
    "step_embed_dim": 16,
    "vf_hidden_dim": 256,
    # PPO
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "clip_eps_scale": 1.0,
    "clip_vf": None,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Hierarchical specific
    "correction_hidden_dim": 128,
    "correction_lr": 3e-4,
    "delta_reg": 10.0,
    "delta_reg_init": None,       # if set, overrides delta_reg as starting value
    "delta_reg_final": 0.1,
    "delta_reg_schedule": "constant",  # "constant", "linear", "exponential"
    "asymmetric_clip": False,
    # Training
    "num_steps": 2048,
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "num_envs": 1,
    "device": "cpu",
}


class PPOHierarchical:
    """PPO with hierarchical advantage estimation.

    Uses a correction network to learn zero-mean per-step adjustments
    on top of the uniform baseline A_t/K.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = {**_DEFAULT_CONFIG, **config}
        self.device = torch.device(self.cfg["device"])

        assert self.cfg["state_dim"] is not None
        assert self.cfg["action_dim"] is not None

        K = self.cfg["K"]

        self.policy = StochasticFlowPolicy(
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            latent_dim=self.cfg["latent_dim"],
            K=K,
            sigma_init=self.cfg["sigma_init"],
            learn_sigma=self.cfg["learn_sigma"],
            sigma_network=self.cfg["sigma_network"],
            step_embed_dim=self.cfg["step_embed_dim"],
        ).to(self.device)

        latent_dim = self.policy.latent_dim

        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        self.correction_net = HierarchicalCorrectionNetwork(
            state_dim=self.cfg["state_dim"],
            latent_dim=latent_dim,
            hidden_dim=self.cfg["correction_hidden_dim"],
            step_embed_dim=self.cfg["step_embed_dim"],
        ).to(self.device)

        # Optimizer with separate LR for correction network
        self.optimizer = optim.Adam([
            {"params": self.policy.parameters(), "lr": self.cfg["lr"]},
            {"params": self.value_fn.parameters(), "lr": self.cfg["lr"]},
            {"params": self.correction_net.parameters(), "lr": self.cfg["correction_lr"]},
        ])
        self._initial_lr = self.cfg["lr"]
        self._initial_corr_lr = self.cfg["correction_lr"]

        self.buffer = RolloutBuffer(
            num_steps=self.cfg["num_steps"],
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            latent_dim=latent_dim,
            K=K,
            num_envs=self.cfg["num_envs"],
            device=self.device,
        )

        self.total_steps = 0

        # Delta-reg annealing state
        self._delta_reg_init = (
            self.cfg["delta_reg_init"]
            if self.cfg["delta_reg_init"] is not None
            else self.cfg["delta_reg"]
        )
        self._delta_reg_final = self.cfg["delta_reg_final"]
        self._delta_reg_schedule = self.cfg["delta_reg_schedule"]
        self._effective_delta_reg = self._delta_reg_init

    # ------------------------------------------------------------------
    # Rollout collection (identical to other agents)
    # ------------------------------------------------------------------

    def collect_rollouts(self, env: Any, num_steps: int) -> Dict[str, float]:
        self.buffer.reset()
        self.policy.eval()
        self.value_fn.eval()

        if not hasattr(self, '_last_obs') or self._last_obs is None:
            obs, _ = env.reset()
        else:
            obs = self._last_obs

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        current_ep_reward = getattr(self, '_current_ep_reward', 0.0)
        current_ep_length = getattr(self, '_current_ep_length', 0)

        with torch.no_grad():
            for _ in range(num_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)

                action, info = self.policy.get_action_and_info(obs_t)
                value = self.value_fn(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                value_np = value.squeeze(0).cpu().item()
                hlp_np = info["holistic_log_prob"].squeeze(0).cpu().numpy()
                pslp_np = info["per_step_log_probs"].squeeze(0).cpu().numpy()
                latents_np = [z.squeeze(0).cpu().numpy() for z in info["latents"]]
                noises_np = [e.squeeze(0).cpu().numpy() for e in info["noises"]]

                action_clipped = np.clip(action_np, -1.0, 1.0)
                next_obs, reward, terminated, truncated, env_info = env.step(action_clipped)
                done = terminated or truncated

                self.buffer.add(
                    state=obs, action=action_clipped, reward=float(reward),
                    done=float(done), value=value_np,
                    holistic_log_prob=hlp_np,
                    per_step_log_probs=pslp_np,
                    latents=latents_np, noises=noises_np,
                )

                current_ep_reward += float(reward)
                current_ep_length += 1
                self.total_steps += 1

                if done:
                    episode_rewards.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)
                    current_ep_reward = 0.0
                    current_ep_length = 0
                    next_obs, _ = env.reset()

                obs = next_obs

        self._last_obs = obs
        self._current_ep_reward = current_ep_reward
        self._current_ep_length = current_ep_length

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            last_value = self.value_fn(obs_t).squeeze(0).cpu().item()

        self.buffer.compute_returns(
            last_value, gamma=self.cfg["gamma"], gae_lambda=self.cfg["gae_lambda"]
        )

        return {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        self.policy.train()
        self.value_fn.train()
        self.correction_net.train()

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropy: list[float] = []
        all_clip_fractions: list[float] = []
        all_approx_kl: list[float] = []
        all_delta_mag: list[float] = []
        all_delta_ratio: list[float] = []
        all_per_step_clip: list[torch.Tensor] = []

        K = self.cfg["K"]
        effective_clip_eps = self.cfg["clip_eps"] * self.cfg["clip_eps_scale"]

        # Asymmetric clipping: tighter upper bound, looser lower bound
        # to address ratio compounding at larger K.
        if self.cfg["asymmetric_clip"]:
            import math
            clip_eps_high = effective_clip_eps / math.sqrt(K)
            clip_eps_low = effective_clip_eps
        else:
            clip_eps_high = None
            clip_eps_low = None

        for epoch in range(self.cfg["num_epochs"]):
            for batch in self.buffer.get_batches(self.cfg["batch_size"]):
                states = batch["states"]
                old_pslp = batch["per_step_log_probs"]
                latents = batch["latents"]
                noises = batch["noises"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Recompute per-step log probs
                new_pslp = self.policy.compute_per_step_log_probs(states, latents, noises)

                # Compute corrections (NOT detached — gradients flow through
                # the policy loss to train the correction network end-to-end)
                delta = self.correction_net(states, latents, K)

                # Hierarchical PPO loss
                policy_loss, info = compute_hierarchical_ppo_loss(
                    new_pslp, old_pslp, advantages, delta, K,
                    clip_eps=effective_clip_eps,
                    clip_eps_high=clip_eps_high,
                    clip_eps_low=clip_eps_low,
                    delta_reg=self._effective_delta_reg,
                )

                # Value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values, clip_vf=self.cfg["clip_vf"]
                )

                # Entropy
                entropy = compute_entropy_bonus(new_pslp)

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg["vf_coef"] * vf_loss
                    - self.cfg["ent_coef"] * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg["max_grad_norm"] is not None:
                    all_params = (
                        list(self.policy.parameters())
                        + list(self.value_fn.parameters())
                        + list(self.correction_net.parameters())
                    )
                    nn.utils.clip_grad_norm_(all_params, self.cfg["max_grad_norm"])
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())
                all_delta_mag.append(info["delta_magnitude"].item())
                all_delta_ratio.append(info["delta_ratio"].item())
                if "per_step_clip_fractions" in info:
                    all_per_step_clip.append(info["per_step_clip_fractions"].detach())

            if self.cfg["target_kl"] is not None:
                mean_kl = float(np.mean(all_approx_kl[-len(all_approx_kl) // max(epoch + 1, 1):]))
                if mean_kl > 1.5 * self.cfg["target_kl"]:
                    break

        if all_per_step_clip:
            mean_per_step_clip = torch.stack(all_per_step_clip).mean(dim=0).cpu().numpy().tolist()
        else:
            mean_per_step_clip = []

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "entropy": float(np.mean(all_entropy)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "approx_kl": float(np.mean(all_approx_kl)),
            "delta_magnitude": float(np.mean(all_delta_mag)),
            "delta_ratio": float(np.mean(all_delta_ratio)),
            "effective_delta_reg": self._effective_delta_reg,
            "per_step_clip_fractions": mean_per_step_clip,
        }

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing."""
        lr = self._initial_lr * (1.0 - progress)
        corr_lr = self._initial_corr_lr * (1.0 - progress)
        self.optimizer.param_groups[0]["lr"] = lr       # policy
        self.optimizer.param_groups[1]["lr"] = lr       # value_fn
        self.optimizer.param_groups[2]["lr"] = corr_lr  # correction_net

    def step_delta_reg(self, progress: float) -> None:
        """Anneal delta_reg based on training progress.

        Args:
            progress: fraction of training complete, in [0, 1).
        """
        schedule = self._delta_reg_schedule
        init = self._delta_reg_init
        final = self._delta_reg_final

        if schedule == "constant":
            self._effective_delta_reg = init
        elif schedule == "linear":
            self._effective_delta_reg = init + (final - init) * progress
        elif schedule == "exponential":
            # init * (final/init)^progress  =  init^(1-progress) * final^progress
            ratio = final / (init + 1e-12)
            self._effective_delta_reg = init * (ratio ** progress)
        else:
            raise ValueError(
                f"Unknown delta_reg_schedule: {schedule}. "
                "Expected 'constant', 'linear', or 'exponential'."
            )
