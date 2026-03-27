"""Rollout buffer for stochastic flow PPO."""

from typing import Dict, Generator, List, Optional

import numpy as np
import torch

from src.algorithms.advantages import compute_gae


class RolloutBuffer:
    """Rollout storage for PPO with stochastic flow policies.

    Stores trajectory data during rollout collection and provides
    random mini-batch iteration for policy updates.

    Args:
        num_steps: Maximum number of environment steps to store.
        state_dim: Observation dimension.
        action_dim: Action dimension.
        latent_dim: Latent space dimension of the flow policy.
        K: Number of denoising steps in the flow.
        num_envs: Number of parallel environments (default 1).
        device: Torch device for mini-batch tensors.
    """

    def __init__(
        self,
        num_steps: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        K: int,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_steps = num_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_envs = num_envs
        self.device = device

        self.pos = 0
        self.full = False
        self._allocate()

    def _allocate(self) -> None:
        """Pre-allocate numpy arrays."""
        n, e = self.num_steps, self.num_envs

        self.states = np.zeros((n, e, self.state_dim) if e > 1 else (n, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((n, e, self.action_dim) if e > 1 else (n, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((n, e) if e > 1 else (n,), dtype=np.float32)
        self.dones = np.zeros((n, e) if e > 1 else (n,), dtype=np.float32)
        self.values = np.zeros((n, e) if e > 1 else (n,), dtype=np.float32)
        self.holistic_log_probs = np.zeros((n, e) if e > 1 else (n,), dtype=np.float32)
        self.per_step_log_probs = np.zeros(
            (n, e, self.K) if e > 1 else (n, self.K), dtype=np.float32
        )

        # Latents: K+1 tensors each of shape (latent_dim,) per step
        # Store as (num_steps, K+1, latent_dim) -- flattened over envs if needed
        self.latents = np.zeros(
            (n, e, self.K + 1, self.latent_dim) if e > 1 else (n, self.K + 1, self.latent_dim),
            dtype=np.float32,
        )
        # Noises: K tensors each of shape (latent_dim,) per step
        self.noises = np.zeros(
            (n, e, self.K, self.latent_dim) if e > 1 else (n, self.K, self.latent_dim),
            dtype=np.float32,
        )

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        done: bool | np.ndarray,
        value: float | np.ndarray,
        holistic_log_prob: float | np.ndarray,
        per_step_log_probs: np.ndarray,
        latents: List[np.ndarray],
        noises: List[np.ndarray],
    ) -> None:
        """Add one environment step to the buffer.

        Args:
            state: Observation, shape (state_dim,) or (num_envs, state_dim).
            action: Action, shape (action_dim,) or (num_envs, action_dim).
            reward: Scalar reward or (num_envs,).
            done: Done flag or (num_envs,).
            value: Value estimate or (num_envs,).
            holistic_log_prob: Holistic log prob or (num_envs,).
            per_step_log_probs: shape (K,) or (num_envs, K).
            latents: List of K+1 arrays, each shape (latent_dim,) or (num_envs, latent_dim).
            noises: List of K arrays, each shape (latent_dim,) or (num_envs, latent_dim).
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.holistic_log_probs[self.pos] = holistic_log_prob
        self.per_step_log_probs[self.pos] = per_step_log_probs

        # Stack latents: (K+1, latent_dim) or (num_envs, K+1, latent_dim)
        self.latents[self.pos] = np.stack(latents, axis=-2)
        self.noises[self.pos] = np.stack(noises, axis=-2)

        self.pos += 1
        if self.pos >= self.num_steps:
            self.full = True

    def compute_returns(
        self,
        last_value: float | np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns.

        Args:
            last_value: Bootstrap value V(s_{T}), scalar or (num_envs,).
            gamma: Discount factor.
            gae_lambda: GAE lambda.
        """
        size = self.pos if not self.full else self.num_steps
        rewards = self.rewards[:size]
        values_arr = self.values[:size]
        dones = self.dones[:size]

        # Append bootstrap value
        last_value = np.asarray(last_value, dtype=np.float32)
        if values_arr.ndim == 1:
            all_values = np.append(values_arr, last_value)
        else:
            all_values = np.concatenate(
                [values_arr, last_value.reshape(1, -1)], axis=0
            )

        self.advantages, self.returns = compute_gae(
            rewards, all_values, dones, gamma, gae_lambda
        )
        # Normalize advantages
        self.advantages = (
            (self.advantages - self.advantages.mean())
            / (self.advantages.std() + 1e-8)
        )

    def get_batches(
        self, batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield random mini-batches of stored transitions as tensors.

        Args:
            batch_size: Number of transitions per mini-batch.

        Yields:
            Dict with keys: states, actions, holistic_log_probs,
            per_step_log_probs, latents (list of K+1 tensors),
            noises (list of K tensors), advantages, returns, values.
        """
        assert self.advantages is not None, "Call compute_returns before get_batches."

        size = self.pos if not self.full else self.num_steps

        # Flatten env dimension if needed
        if self.num_envs > 1:
            total = size * self.num_envs
            states = self.states[:size].reshape(total, self.state_dim)
            actions = self.actions[:size].reshape(total, self.action_dim)
            hlp = self.holistic_log_probs[:size].reshape(total)
            pslp = self.per_step_log_probs[:size].reshape(total, self.K)
            lat = self.latents[:size].reshape(total, self.K + 1, self.latent_dim)
            noi = self.noises[:size].reshape(total, self.K, self.latent_dim)
            adv = self.advantages[:size].reshape(total)
            ret = self.returns[:size].reshape(total)
            val = self.values[:size].reshape(total)
        else:
            total = size
            states = self.states[:size]
            actions = self.actions[:size]
            hlp = self.holistic_log_probs[:size]
            pslp = self.per_step_log_probs[:size]
            lat = self.latents[:size]
            noi = self.noises[:size]
            adv = self.advantages[:size]
            ret = self.returns[:size]
            val = self.values[:size]

        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            if end > total:
                break  # drop incomplete last batch

            idx = indices[start:end]
            b_lat = torch.tensor(lat[idx], dtype=torch.float32, device=self.device)
            b_noi = torch.tensor(noi[idx], dtype=torch.float32, device=self.device)

            # Convert latents to list of K+1 tensors, noises to list of K tensors
            latents_list = [b_lat[:, i, :] for i in range(self.K + 1)]
            noises_list = [b_noi[:, i, :] for i in range(self.K)]

            yield {
                "states": torch.tensor(states[idx], dtype=torch.float32, device=self.device),
                "actions": torch.tensor(actions[idx], dtype=torch.float32, device=self.device),
                "holistic_log_probs": torch.tensor(hlp[idx], dtype=torch.float32, device=self.device),
                "per_step_log_probs": torch.tensor(pslp[idx], dtype=torch.float32, device=self.device),
                "latents": latents_list,
                "noises": noises_list,
                "advantages": torch.tensor(adv[idx], dtype=torch.float32, device=self.device),
                "returns": torch.tensor(ret[idx], dtype=torch.float32, device=self.device),
                "values": torch.tensor(val[idx], dtype=torch.float32, device=self.device),
            }

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self.pos = 0
        self.full = False
        self.advantages = None
        self.returns = None
