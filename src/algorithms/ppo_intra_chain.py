"""PPO agent with intra-chain temporal credit assignment (novel algorithm).

Instead of splitting the environment advantage with arbitrary weights,
this agent learns a value function V(s, z_k, k) over the internal
generative chain. Per-step advantages are computed via temporal
differencing within the chain:

    A_{t,k} = V(s, z_{k-1}, k-1) - V(s, z_k, k)

This provides genuine per-step credit signals grounded in the
environment reward, rather than heuristic weight splitting.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.buffers import RolloutBuffer
from src.algorithms.losses import (
    compute_entropy_bonus,
    compute_per_step_ppo_loss,
    compute_value_loss,
)
from src.models.stochastic_flow_policy import StochasticFlowPolicy
from src.models.value_function import IntraChainValueFunction, ValueFunction


_DEFAULT_CONFIG: Dict[str, Any] = {
    # Environment
    "state_dim": None,
    "action_dim": None,
    # Policy architecture
    "hidden_dim": 256,
    "latent_dim": None,
    "K": 4,
    "sigma_init": 1.0,
    "learn_sigma": False,
    "sigma_network": False,
    "step_embed_dim": 16,
    # Value function
    "vf_hidden_dim": 256,
    # PPO hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "clip_eps_scale": 1.0,
    "clip_vf": None,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Intra-chain specific
    "intra_vf_coef": 0.5,   # loss weight for intra-chain value function
    "intra_vf_lr": 3e-4,    # separate LR for intra-chain VF
    "intra_hidden_dim": 128, # hidden dim for intra-chain VF (smaller)
    # Training
    "num_steps": 2048,
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "num_envs": 1,
    "device": "cpu",
}


class PPOIntraChain:
    """PPO agent with intra-chain temporal credit assignment.

    Uses a learned value function V(s, z_k, k) to compute genuine
    per-step advantages via temporal differencing within the generative
    chain, rather than splitting the environment advantage with weights.

    Args:
        config: Dict of hyperparameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = {**_DEFAULT_CONFIG, **config}
        self.device = torch.device(self.cfg["device"])

        assert self.cfg["state_dim"] is not None
        assert self.cfg["action_dim"] is not None

        K = self.cfg["K"]

        # Build policy
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

        # Environment value function V(s)
        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        # Intra-chain value function V(s, z_k, k)
        self.intra_vf = IntraChainValueFunction(
            state_dim=self.cfg["state_dim"],
            latent_dim=latent_dim,
            K=K,
            hidden_dim=self.cfg["intra_hidden_dim"],
            step_embed_dim=self.cfg["step_embed_dim"],
        ).to(self.device)

        # Optimizer with separate param groups
        self.optimizer = optim.Adam([
            {"params": self.policy.parameters(), "lr": self.cfg["lr"]},
            {"params": self.value_fn.parameters(), "lr": self.cfg["lr"]},
            {"params": self.intra_vf.parameters(), "lr": self.cfg["intra_vf_lr"]},
        ])
        self._initial_lr = self.cfg["lr"]
        self._initial_intra_lr = self.cfg["intra_vf_lr"]

        # Rollout buffer
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

    # ------------------------------------------------------------------
    # Rollout collection (same as other agents)
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
                    state=obs,
                    action=action_clipped,
                    reward=float(reward),
                    done=float(done),
                    value=value_np,
                    holistic_log_prob=hlp_np,
                    per_step_log_probs=pslp_np,
                    latents=latents_np,
                    noises=noises_np,
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
        self.intra_vf.train()

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_intra_vf_losses: list[float] = []
        all_entropy: list[float] = []
        all_clip_fractions: list[float] = []
        all_approx_kl: list[float] = []
        all_per_step_clip: list[torch.Tensor] = []

        for epoch in range(self.cfg["num_epochs"]):
            for batch in self.buffer.get_batches(self.cfg["batch_size"]):
                states = batch["states"]
                old_pslp = batch["per_step_log_probs"]
                latents = batch["latents"]
                noises = batch["noises"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Recompute per-step log probs under current policy
                new_pslp = self.policy.compute_per_step_log_probs(states, latents, noises)

                # Compute intra-chain per-step advantages (detached so
                # policy loss doesn't backprop through intra_vf — the intra_vf
                # is trained only by its own MSE loss)
                intra_adv = self.intra_vf.compute_intra_advantages(
                    states, latents, advantages
                ).detach()

                # Per-step policy loss using intra-chain advantages directly
                effective_clip_eps = self.cfg["clip_eps"] * self.cfg["clip_eps_scale"]
                policy_loss, info = self._compute_intra_policy_loss(
                    new_pslp, old_pslp, intra_adv, effective_clip_eps
                )

                # Environment value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values, clip_vf=self.cfg["clip_vf"]
                )

                # Intra-chain value function loss
                intra_vf_loss = self.intra_vf.compute_loss(states, latents, advantages)

                # Entropy bonus
                entropy = compute_entropy_bonus(new_pslp)

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg["vf_coef"] * vf_loss
                    + self.cfg["intra_vf_coef"] * intra_vf_loss
                    - self.cfg["ent_coef"] * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg["max_grad_norm"] is not None:
                    all_params = (
                        list(self.policy.parameters())
                        + list(self.value_fn.parameters())
                        + list(self.intra_vf.parameters())
                    )
                    nn.utils.clip_grad_norm_(all_params, self.cfg["max_grad_norm"])
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(vf_loss.item())
                all_intra_vf_losses.append(intra_vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())
                if "per_step_clip_fractions" in info:
                    all_per_step_clip.append(info["per_step_clip_fractions"].detach())

            # Early stopping on KL
            if self.cfg["target_kl"] is not None:
                mean_kl = float(np.mean(all_approx_kl[-len(all_approx_kl) // max(epoch + 1, 1):]))
                if mean_kl > 1.5 * self.cfg["target_kl"]:
                    break

        # Per-step clip fractions
        if all_per_step_clip:
            mean_per_step_clip = torch.stack(all_per_step_clip).mean(dim=0).cpu().numpy().tolist()
        else:
            mean_per_step_clip = []

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "intra_vf_loss": float(np.mean(all_intra_vf_losses)),
            "entropy": float(np.mean(all_entropy)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "approx_kl": float(np.mean(all_approx_kl)),
            "per_step_clip_fractions": mean_per_step_clip,
        }

    def _compute_intra_policy_loss(
        self,
        step_log_probs_new: torch.Tensor,
        step_log_probs_old: torch.Tensor,
        intra_advantages: torch.Tensor,
        clip_eps: float,
    ):
        """PPO clipped loss with intra-chain advantages.

        Unlike the weight-based loss, here each step gets its own
        advantage value from the intra-chain value function, not
        a fraction of the environment advantage.

        Args:
            step_log_probs_new: shape (batch, K).
            step_log_probs_old: shape (batch, K).
            intra_advantages: shape (batch, K), per-step advantages.
            clip_eps: clipping epsilon.

        Returns:
            loss: scalar.
            info: dict with diagnostics.
        """
        log_ratios = step_log_probs_new - step_log_probs_old
        log_ratios = torch.clamp(log_ratios, -20.0, 20.0)
        ratios = torch.exp(log_ratios)

        surr1 = ratios * intra_advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * intra_advantages

        loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

        with torch.no_grad():
            per_step_clip_fractions = (
                (torch.abs(ratios - 1.0) > clip_eps).float().mean(dim=0)
            )
            clip_fraction = per_step_clip_fractions.mean()
            approx_kl = ((ratios - 1) - log_ratios).mean()
            mean_ratio = ratios.mean()

        info = {
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "mean_ratio": mean_ratio,
            "per_step_clip_fractions": per_step_clip_fractions,
        }
        return loss, info

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing."""
        lr = self._initial_lr * (1.0 - progress)
        intra_lr = self._initial_intra_lr * (1.0 - progress)
        self.optimizer.param_groups[0]["lr"] = lr       # policy
        self.optimizer.param_groups[1]["lr"] = lr       # value_fn
        self.optimizer.param_groups[2]["lr"] = intra_lr  # intra_vf
