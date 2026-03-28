"""PPO agent with KL budget waterfilling across flow steps.

The key insight from channel capacity theory: the optimal allocation of a
total KL budget D across K denoising steps is

    d_k = D * alpha_k^2 / sum_j alpha_j^2

where alpha_k is the per-step advantage magnitude (proxied by observed
per-step KL divergence).  Each step's clip epsilon is then set to
eps_k = sqrt(2 * d_k), giving steps that the policy is changing most
a wider trust region and steps that are stable a tighter one.

This is adaptive: each PPO epoch recomputes the allocation from the
current per-step KL, so the budget naturally flows to where it is needed.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.buffers import RolloutBuffer
from src.algorithms.losses import (
    compute_entropy_bonus,
    compute_value_loss,
    compute_waterfill_ppo_loss,
)
from src.models.stochastic_flow_policy import StochasticFlowPolicy
from src.models.value_function import ValueFunction


_DEFAULT_CONFIG: Dict[str, Any] = {
    # Environment
    "state_dim": None,        # required
    "action_dim": None,       # required
    # Policy architecture
    "hidden_dim": 256,
    "latent_dim": None,  # defaults to action_dim
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
    "clip_eps": 0.2,         # not used directly; total_kl_budget controls clipping
    "clip_vf": None,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Waterfilling parameters
    "total_kl_budget": 0.02,  # total KL budget D across all steps
    "min_budget_frac": 0.01,  # minimum fraction of D per step
    # Training
    "num_steps": 2048,
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "num_envs": 1,
    "device": "cpu",
}


class PPOWaterfill:
    """PPO agent with KL budget waterfilling across flow steps.

    Allocates per-step clip epsilon adaptively based on observed per-step KL
    divergence, following the waterfilling solution from information theory.

    Args:
        config: Dict of hyperparameters. See _DEFAULT_CONFIG for keys.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = {**_DEFAULT_CONFIG, **config}
        self.device = torch.device(self.cfg["device"])

        assert self.cfg["state_dim"] is not None, "state_dim required"
        assert self.cfg["action_dim"] is not None, "action_dim required"

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

        # Build value function
        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        # Optimizer
        all_params = list(self.policy.parameters()) + list(self.value_fn.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.cfg["lr"])
        self._initial_lr = self.cfg["lr"]

        # Rollout buffer — with vec envs, allocate for num_steps * num_envs
        n_envs = self.cfg.get("num_envs", 1)
        self.buffer = RolloutBuffer(
            num_steps=self.cfg["num_steps"] * n_envs,
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            latent_dim=latent_dim,
            K=K,
            num_envs=1,  # flattened storage
            device=self.device,
        )

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Rollout collection (uses shared rollout infrastructure)
    # ------------------------------------------------------------------

    def collect_rollouts(self, env: Any, num_steps: int) -> Dict[str, float]:
        """Collect environment transitions and fill the rollout buffer.

        Supports both single and vectorized environments.

        Args:
            env: Gymnasium-compatible environment (single or vectorized).
            num_steps: Number of steps per environment to collect.

        Returns:
            Dict with rollout statistics.
        """
        from src.algorithms.rollout import collect_rollouts_vec

        # Build mutable state dict (shared across calls)
        if not hasattr(self, '_rollout_state'):
            self._rollout_state = {
                '_last_obs': None,
                '_current_ep_rewards': None,
                '_current_ep_lengths': None,
                'total_steps': self.total_steps,
            }
        self._rollout_state['total_steps'] = self.total_steps

        stats = collect_rollouts_vec(
            env=env,
            policy=self.policy,
            value_fn=self.value_fn,
            buffer=self.buffer,
            num_steps=num_steps,
            device=self.device,
            agent_state=self._rollout_state,
            gamma=self.cfg["gamma"],
            gae_lambda=self.cfg["gae_lambda"],
        )

        self.total_steps = self._rollout_state['total_steps']
        self._last_obs = self._rollout_state['_last_obs']
        return stats

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run multiple epochs of waterfilling PPO updates on the buffer.

        Returns:
            Dict with training statistics.
        """
        self.policy.train()
        self.value_fn.train()

        K = self.cfg["K"]
        total_kl_budget = self.cfg["total_kl_budget"]
        min_budget_frac = self.cfg["min_budget_frac"]

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropy: list[float] = []
        all_clip_fractions: list[float] = []
        all_approx_kl: list[float] = []
        all_per_step_clip: list[torch.Tensor] = []
        all_per_step_eps: list[torch.Tensor] = []
        all_per_step_budgets: list[torch.Tensor] = []

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
                new_pslp = self.policy.compute_per_step_log_probs(
                    states, latents, noises
                )

                # Waterfilling policy loss
                policy_loss, info = compute_waterfill_ppo_loss(
                    new_pslp, old_pslp, advantages,
                    K=K,
                    total_kl_budget=total_kl_budget,
                    min_budget_frac=min_budget_frac,
                )

                # Value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values,
                    clip_vf=self.cfg["clip_vf"],
                )

                # Entropy bonus
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
                    )
                    nn.utils.clip_grad_norm_(all_params, self.cfg["max_grad_norm"])
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())
                all_per_step_clip.append(info["per_step_clip_fractions"].detach())
                all_per_step_eps.append(info["per_step_eps"].detach())
                all_per_step_budgets.append(info["per_step_budgets"].detach())

            # Early stopping on KL divergence
            if self.cfg["target_kl"] is not None:
                recent_kl = all_approx_kl[
                    -len(all_approx_kl) // max(epoch + 1, 1):
                ]
                if recent_kl and float(np.mean(recent_kl)) > 1.5 * self.cfg["target_kl"]:
                    break

        # Average diagnostics
        if all_per_step_clip:
            mean_per_step_clip = (
                torch.stack(all_per_step_clip).mean(dim=0).cpu().numpy().tolist()
            )
        else:
            mean_per_step_clip = []

        if all_per_step_eps:
            mean_per_step_eps = (
                torch.stack(all_per_step_eps).mean(dim=0).cpu().numpy().tolist()
            )
        else:
            mean_per_step_eps = []

        if all_per_step_budgets:
            mean_per_step_budgets = (
                torch.stack(all_per_step_budgets).mean(dim=0).cpu().numpy().tolist()
            )
        else:
            mean_per_step_budgets = []

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "entropy": float(np.mean(all_entropy)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "approx_kl": float(np.mean(all_approx_kl)),
            "per_step_clip_fractions": mean_per_step_clip,
            "per_step_eps": mean_per_step_eps,
            "per_step_budgets": mean_per_step_budgets,
        }

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing. Call with progress in [0, 1]."""
        lr = self._initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
