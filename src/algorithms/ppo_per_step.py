"""PPO agent using per-step credit assignment (modes 2/3/4)."""

import math
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
    compute_weight_entropy,
)
from src.models.stochastic_flow_policy import StochasticFlowPolicy
from src.models.value_function import ValueFunction
from src.models.weighting_network import (
    DenoisingDiscountWeights,
    KLInverseWeights,
    LearnedGlobalWeights,
    StateDependentWeights,
    UniformWeights,
)


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
    # Weighting
    "weighting_mode": "uniform",  # "uniform", "learned_global", "state_dependent", "kl_inverse"
    "weight_hidden_dim": 128,
    "weight_step_embed_dim": 16,
    "kl_beta": 5.0,
    "learn_kl_beta": False,
    # PPO hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "clip_vf": None,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "weight_ent_coef": 0.01,  # entropy regularizer on learned weights
    "clip_eps_scale": 1.0,    # multiply clip_eps for per-step (>1 = less conservative)
    "asymmetric_clip": False, # use asymmetric per-step clipping
    "clip_eps_high": None,    # upper clip override (auto-computed when asymmetric_clip=True)
    "clip_eps_low": None,     # lower clip override (auto-computed when asymmetric_clip=True)
    # Training
    "num_steps": 2048,
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_lr": 3e-4,
    "num_envs": 1,
    "device": "cpu",
}


class PPOPerStep:
    """PPO agent with per-step credit assignment.

    Supports three weighting modes:
      - "uniform": 1/K credit per step (mode 2).
      - "learned_global": softmax-normalized learned vector (mode 3).
      - "state_dependent": MLP-based weights conditioned on (state, z_k, k) (mode 4).

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

        latent_dim = self.policy.latent_dim  # resolved (may default to action_dim)

        # Build value function
        self.value_fn = ValueFunction(
            state_dim=self.cfg["state_dim"],
            hidden_dim=self.cfg["vf_hidden_dim"],
        ).to(self.device)

        # Build weighting network
        mode = self.cfg["weighting_mode"]
        if mode == "uniform":
            self.weight_net: nn.Module = UniformWeights().to(self.device)
        elif mode == "learned_global":
            self.weight_net = LearnedGlobalWeights(K=K).to(self.device)
        elif mode == "state_dependent":
            self.weight_net = StateDependentWeights(
                latent_dim=latent_dim,
                hidden_dim=self.cfg["weight_hidden_dim"],
                step_embed_dim=self.cfg["weight_step_embed_dim"],
            ).to(self.device)
        elif mode == "kl_inverse":
            self.weight_net = KLInverseWeights(
                beta=self.cfg["kl_beta"],
                learn_beta=self.cfg["learn_kl_beta"],
            ).to(self.device)
        elif mode == "denoising_discount":
            self.weight_net = DenoisingDiscountWeights(
                gamma_denoise=self.cfg.get("gamma_denoise", 0.95),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown weighting_mode: {mode}")

        # Optimizers
        policy_vf_params = list(self.policy.parameters()) + list(self.value_fn.parameters())
        weight_params = list(self.weight_net.parameters())

        if weight_params:
            self.optimizer = optim.Adam(
                [
                    {"params": policy_vf_params, "lr": self.cfg["lr"]},
                    {"params": weight_params, "lr": self.cfg["weight_lr"]},
                ]
            )
        else:
            self.optimizer = optim.Adam(policy_vf_params, lr=self.cfg["lr"])
        self._initial_lr = self.cfg["lr"]
        self._initial_weight_lr = self.cfg.get("weight_lr", self.cfg["lr"])

        # Rollout buffer (use policy's resolved latent_dim)
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
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, env: Any, num_steps: int) -> Dict[str, float]:
        """Collect environment transitions and fill the rollout buffer.

        Args:
            env: Gymnasium-compatible environment.
            num_steps: Number of steps to collect.

        Returns:
            Dict with rollout statistics.
        """
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

        stats = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }
        return stats

    # ------------------------------------------------------------------
    # Compute weights
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        states: torch.Tensor,
        latents: list[torch.Tensor],
        step_log_probs_new: Optional[torch.Tensor] = None,
        step_log_probs_old: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-step credit weights.

        Args:
            states: shape (batch, state_dim).
            latents: list of K+1 tensors, each (batch, latent_dim).
            step_log_probs_new: Per-step log probs under current policy, (batch, K).
                Required for kl_inverse mode.
            step_log_probs_old: Per-step log probs under old policy, (batch, K).
                Required for kl_inverse mode.

        Returns:
            weights: shape (batch, K), summing to 1.
        """
        mode = self.cfg["weighting_mode"]
        K = self.cfg["K"]

        if mode == "uniform":
            return self.weight_net(states, latents, K)
        elif mode == "learned_global":
            return self.weight_net(states, latents, K)
        elif mode == "state_dependent":
            # State-dependent weights need encoded state
            s_enc = self.policy._encode_state(states)
            return self.weight_net(s_enc, latents, K)
        elif mode == "kl_inverse":
            return self.weight_net(
                states, latents, K,
                step_log_probs_new=step_log_probs_new,
                step_log_probs_old=step_log_probs_old,
            )
        elif mode == "denoising_discount":
            return self.weight_net(states, latents, K)
        else:
            raise ValueError(f"Unknown weighting_mode: {mode}")

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run multiple epochs of per-step PPO updates on the buffer.

        Returns:
            Dict with training statistics.
        """
        self.policy.train()
        self.value_fn.train()
        self.weight_net.train()

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
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

                # Compute credit weights
                weights = self._compute_weights(
                    states, latents,
                    step_log_probs_new=new_pslp,
                    step_log_probs_old=old_pslp,
                )

                # Policy loss (scale clip_eps for per-step to reduce conservatism)
                effective_clip_eps = self.cfg["clip_eps"] * self.cfg["clip_eps_scale"]

                # Asymmetric clipping: tighter upper bound to prevent ratio
                # explosion compounding across K steps.
                loss_kwargs: Dict[str, Any] = {}
                if self.cfg["asymmetric_clip"]:
                    K = self.cfg["K"]
                    loss_kwargs["clip_eps_high"] = effective_clip_eps / math.sqrt(K)
                    loss_kwargs["clip_eps_low"] = effective_clip_eps

                policy_loss, info = compute_per_step_ppo_loss(
                    new_pslp, old_pslp, advantages, weights,
                    clip_eps=effective_clip_eps, **loss_kwargs,
                )

                # Value loss
                new_values = self.value_fn(states)
                vf_loss = compute_value_loss(
                    new_values, returns, old_values, clip_vf=self.cfg["clip_vf"]
                )

                # Entropy bonus
                entropy = compute_entropy_bonus(new_pslp)

                # Weight entropy regularizer (prevents collapse to one-hot)
                weight_ent = compute_weight_entropy(weights)

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg["vf_coef"] * vf_loss
                    - self.cfg["ent_coef"] * entropy
                    - self.cfg["weight_ent_coef"] * weight_ent
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg["max_grad_norm"] is not None:
                    all_params = (
                        list(self.policy.parameters())
                        + list(self.value_fn.parameters())
                        + list(self.weight_net.parameters())
                    )
                    nn.utils.clip_grad_norm_(all_params, self.cfg["max_grad_norm"])
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy.append(entropy.item())
                all_clip_fractions.append(info["clip_fraction"].item())
                all_approx_kl.append(info["approx_kl"].item())
                all_per_step_clip.append(info["per_step_clip_fractions"].detach())

            # Early stopping on KL divergence
            if self.cfg["target_kl"] is not None:
                recent_kl = all_approx_kl[-len(all_approx_kl) // max(epoch + 1, 1):]
                if recent_kl and float(np.mean(recent_kl)) > 1.5 * self.cfg["target_kl"]:
                    break

        # Average per-step clip fractions
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
            "per_step_clip_fractions": mean_per_step_clip,
        }

    def step_lr(self, progress: float) -> None:
        """Linear LR annealing. Call with progress in [0, 1]."""
        lr = self._initial_lr * (1.0 - progress)
        weight_lr = self._initial_weight_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            if "weight" in str(param_group.get("name", "")):
                param_group["lr"] = weight_lr
            else:
                param_group["lr"] = lr
        # Fallback: if no named groups, just set all to policy lr
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]["lr"] = lr
        elif len(self.optimizer.param_groups) == 2:
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = weight_lr

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
                weight_info = ""
                if self.cfg["weighting_mode"] == "learned_global":
                    with torch.no_grad():
                        w = torch.softmax(self.weight_net.alpha, dim=0)
                    weight_info = f" weights={w.cpu().numpy().round(3)}"

                print(
                    f"[Iter {iteration}/{num_iterations}] "
                    f"steps={self.total_steps} "
                    f"reward={rollout_stats['mean_reward']:.2f} "
                    f"pi_loss={update_stats['policy_loss']:.4f} "
                    f"vf_loss={update_stats['value_loss']:.4f} "
                    f"entropy={update_stats['entropy']:.4f} "
                    f"kl={update_stats['approx_kl']:.4f}"
                    f"{weight_info}"
                )

            if callback is not None:
                callback(locals(), globals())

        return history
