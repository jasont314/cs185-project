"""PPO agent using holistic (standard) clipped surrogate loss."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.buffers import RolloutBuffer
from src.algorithms.losses import (
    compute_entropy_bonus,
    compute_holistic_ppo_loss,
    compute_value_loss,
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
    "clip_eps": 0.2,
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


class PPOHolistic:
    """PPO agent with holistic (mode 1) policy loss.

    Uses a StochasticFlowPolicy and ValueFunction. Collects rollouts,
    computes GAE, and performs multiple epochs of mini-batch updates
    using the standard clipped surrogate objective.

    Args:
        config: Dict of hyperparameters. See _DEFAULT_CONFIG for keys.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = {**_DEFAULT_CONFIG, **config}
        self.device = torch.device(self.cfg["device"])

        assert self.cfg["state_dim"] is not None, "state_dim required"
        assert self.cfg["action_dim"] is not None, "action_dim required"

        # Build networks
        self.policy = StochasticFlowPolicy(
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            latent_dim=self.cfg["latent_dim"],
            K=self.cfg["K"],
            sigma_init=self.cfg["sigma_init"],
            learn_sigma=self.cfg["learn_sigma"],
            sigma_network=self.cfg["sigma_network"],
            step_embed_dim=self.cfg["step_embed_dim"],
        ).to(self.device)

        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        # torch.compile for faster forward/backward (PyTorch 2.0+)
        # Disabled by default — enable with --compile true on supported systems
        if self.cfg.get("compile", False):
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                self.value_fn = torch.compile(self.value_fn, mode="reduce-overhead")
            except Exception:
                pass  # graceful fallback if compile not supported

        # Optimizer + linear LR annealing
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()),
            lr=self.cfg["lr"],
        )
        self._initial_lr = self.cfg["lr"]

        # Rollout buffer — with vec envs, each step produces num_envs
        # transitions, so allocate accordingly (stored flattened, num_envs=1)
        n_envs = self.cfg.get("num_envs", 1)
        self.buffer = RolloutBuffer(
            num_steps=self.cfg["num_steps"] * n_envs,
            state_dim=self.cfg["state_dim"],
            action_dim=self.cfg["action_dim"],
            latent_dim=self.policy.latent_dim,
            K=self.cfg["K"],
            num_envs=1,  # flattened storage
            device=self.device,
        )

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Rollout collection
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
        # Backward compat aliases
        self._last_obs = self._rollout_state['_last_obs']
        return stats

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run multiple epochs of PPO updates on the buffer.

        Returns:
            Dict with training statistics.
        """
        self.policy.train()
        self.value_fn.train()

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropy: list[float] = []
        all_clip_fractions: list[float] = []
        all_approx_kl: list[float] = []

        for epoch in range(self.cfg["num_epochs"]):
            for batch in self.buffer.get_batches(self.cfg["batch_size"]):
                states = batch["states"]
                old_hlp = batch["holistic_log_probs"]
                latents = batch["latents"]
                noises = batch["noises"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Recompute log probs under current policy
                new_hlp = self.policy.compute_holistic_log_prob(states, latents, noises)
                new_pslp = self.policy.compute_per_step_log_probs(states, latents, noises)

                # Policy loss
                policy_loss, info = compute_holistic_ppo_loss(
                    new_hlp, old_hlp, advantages, clip_eps=self.cfg["clip_eps"]
                )

                # Value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values, clip_vf=self.cfg["clip_vf"]
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
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.value_fn.parameters()),
                        self.cfg["max_grad_norm"],
                    )
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())

            # Early stopping on KL divergence
            if self.cfg["target_kl"] is not None:
                mean_kl = float(np.mean(all_approx_kl[-len(all_approx_kl) // max(epoch + 1, 1):]))
                if mean_kl > 1.5 * self.cfg["target_kl"]:
                    break

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "entropy": float(np.mean(all_entropy)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "approx_kl": float(np.mean(all_approx_kl)),
        }

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing. Call with progress in [0, 1]."""
        lr = self._initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: Any,
        total_timesteps: int,
        log_interval: int = 1,
        callback: Optional[Any] = None,
    ) -> Dict[str, list]:
        """Run the full PPO training loop.

        Args:
            env: Gymnasium-compatible environment.
            total_timesteps: Total environment steps to train for.
            log_interval: Print stats every this many iterations.
            callback: Optional callable(locals, globals) called each iteration.

        Returns:
            Dict of logged statistics lists.
        """
        self._last_obs = None
        self._current_ep_reward = 0.0
        self._current_ep_length = 0

        num_iterations = total_timesteps // self.cfg["num_steps"]
        history: Dict[str, list] = {
            "mean_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
        }

        for iteration in range(1, num_iterations + 1):
            rollout_stats = self.collect_rollouts(env, self.cfg["num_steps"])
            update_stats = self.update()

            history["mean_reward"].append(rollout_stats["mean_reward"])
            for k in ["policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl"]:
                history[k].append(update_stats[k])

            if log_interval > 0 and iteration % log_interval == 0:
                print(
                    f"[Iter {iteration}/{num_iterations}] "
                    f"steps={self.total_steps} "
                    f"reward={rollout_stats['mean_reward']:.2f} "
                    f"pi_loss={update_stats['policy_loss']:.4f} "
                    f"vf_loss={update_stats['value_loss']:.4f} "
                    f"entropy={update_stats['entropy']:.4f} "
                    f"kl={update_stats['approx_kl']:.4f}"
                )

            if callback is not None:
                callback(locals(), globals())

        return history
