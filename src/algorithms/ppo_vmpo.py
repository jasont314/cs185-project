"""PPO agent using Per-Step V-MPO loss (softmax-weighted MLE, no importance ratios)."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.buffers import RolloutBuffer
from src.algorithms.losses import (
    compute_entropy_bonus,
    compute_value_loss,
    compute_vmpo_loss,
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
    # V-MPO hyperparams
    "eps_eta": 0.01,          # KL constraint target for temperature dual
    "top_frac": 0.5,          # fraction of batch to keep by advantage
    "eta_lr": 1e-2,           # learning rate for temperature parameters
    # PPO / shared hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_vf": None,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Training
    "num_steps": 2048,
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "num_envs": 1,
    "device": "cpu",
}


class PPOVmpo:
    """Per-Step V-MPO agent.

    Replaces PPO's clipped importance ratios with softmax-weighted maximum
    likelihood at each flow step, using learned per-step temperatures.
    This eliminates ratio compounding entirely.

    Args:
        config: Dict of hyperparameters. See _DEFAULT_CONFIG for keys.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = {**_DEFAULT_CONFIG, **config}
        self.device = torch.device(self.cfg["device"])

        assert self.cfg["state_dim"] is not None, "state_dim required"
        assert self.cfg["action_dim"] is not None, "action_dim required"

        K = self.cfg["K"]

        # Build networks
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

        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        # Learnable per-step log-temperatures (one per flow step)
        self.log_etas = nn.Parameter(
            torch.zeros(K, device=self.device)
        )

        # torch.compile for faster forward/backward (PyTorch 2.0+)
        if self.cfg.get("compile", False):
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                self.value_fn = torch.compile(self.value_fn, mode="reduce-overhead")
            except Exception:
                pass

        # Main optimizer for policy + value function
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()),
            lr=self.cfg["lr"],
        )
        self._initial_lr = self.cfg["lr"]

        # Separate optimizer for temperature parameters (often needs different LR)
        self.eta_optimizer = optim.Adam(
            [self.log_etas],
            lr=self.cfg["eta_lr"],
        )
        self._initial_eta_lr = self.cfg["eta_lr"]

        # Rollout buffer
        n_envs = self.cfg.get("num_envs", 1)
        self.buffer = RolloutBuffer(
            num_steps=self.cfg["num_steps"] * n_envs,
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            latent_dim=self.policy.latent_dim,
            K=K,
            num_envs=1,  # flattened storage
            device=self.device,
        )

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Rollout collection (shared implementation)
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
        """Run multiple epochs of V-MPO updates on the buffer.

        Returns:
            Dict with training statistics.
        """
        self.policy.train()
        self.value_fn.train()

        K = self.cfg["K"]

        all_policy_losses: list[float] = []
        all_eta_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropy: list[float] = []
        all_clip_fractions: list[float] = []
        all_approx_kl: list[float] = []

        for epoch in range(self.cfg["num_epochs"]):
            for batch in self.buffer.get_batches(self.cfg["batch_size"]):
                states = batch["states"]
                latents = batch["latents"]
                noises = batch["noises"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Recompute per-step log probs under current policy
                new_pslp = self.policy.compute_per_step_log_probs(
                    states, latents, noises
                )

                # V-MPO policy loss (no importance ratios)
                policy_loss, eta_loss, info = compute_vmpo_loss(
                    step_log_probs_new=new_pslp,
                    advantages=advantages,
                    K=K,
                    log_etas=self.log_etas,
                    eps_eta=self.cfg["eps_eta"],
                    top_frac=self.cfg["top_frac"],
                )

                # Value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values,
                    clip_vf=self.cfg["clip_vf"],
                )

                # Entropy bonus
                entropy = compute_entropy_bonus(new_pslp)

                # Total loss for policy + value (eta is optimised separately)
                loss = (
                    policy_loss
                    + self.cfg["vf_coef"] * vf_loss
                    - self.cfg["ent_coef"] * entropy
                )

                # Update policy + value function
                self.optimizer.zero_grad()
                self.eta_optimizer.zero_grad()

                # Backward on combined loss (eta_loss needs grad for log_etas)
                total_loss = loss + eta_loss
                total_loss.backward()

                if self.cfg["max_grad_norm"] is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters())
                        + list(self.value_fn.parameters()),
                        self.cfg["max_grad_norm"],
                    )

                self.optimizer.step()
                self.eta_optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_eta_losses.append(eta_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())

        # Build eta diagnostics
        etas_final = [torch.exp(self.log_etas[k]).item() for k in range(K)]

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "entropy": float(np.mean(all_entropy)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "approx_kl": float(np.mean(all_approx_kl)),
            "eta_loss": float(np.mean(all_eta_losses)),
            "etas": etas_final,
        }

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing. Call with progress in [0, 1]."""
        lr = self._initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        eta_lr = self._initial_eta_lr * (1.0 - progress)
        for param_group in self.eta_optimizer.param_groups:
            param_group["lr"] = eta_lr
