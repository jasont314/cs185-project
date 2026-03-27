"""MLP value functions for PPO."""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to a linear layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ValueFunction(nn.Module):
    """Simple MLP state-value function V(s).

    Architecture: state_dim -> 256 -> 256 -> 1
    Uses orthogonal initialization with sqrt(2) gain for hidden layers
    and gain=1.0 for the output layer.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Hidden layers with sqrt(2) gain (common for tanh activations)
        for layer in [self.net[0], self.net[2]]:
            _orthogonal_init(layer, gain=2.0 ** 0.5)
        # Output layer with small gain
        _orthogonal_init(self.net[4], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            state: Observation tensor, shape (batch, state_dim).

        Returns:
            Value estimate, shape (batch,).
        """
        return self.net(state).squeeze(-1)


class SinusoidalStepEmbedding(nn.Module):
    """Sinusoidal embedding for step index (shared with policy)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        k = k.float()
        if k.dim() == 0:
            k = k.unsqueeze(0)
        half = self.embed_dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=k.device, dtype=torch.float32) / half
        )
        args = k.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class IntraChainValueFunction(nn.Module):
    """Value function over the internal generative chain: V(s, z_k, k).

    Estimates the "action quality contribution" at each internal step.
    Used to compute genuine per-step advantages via temporal differencing
    within the generative chain:

        A_{t,k} = V(s, z_{k-1}, k-1) - V(s, z_k, k)

    Anchored so V(s, z_K, K) = 0 (pure noise has no value).

    Args:
        state_dim: Observation dimension.
        latent_dim: Latent z dimension (= action_dim).
        K: Number of flow steps.
        hidden_dim: Hidden layer width.
        step_embed_dim: Dimension of step index embedding.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        K: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.step_embed = SinusoidalStepEmbedding(step_embed_dim)

        input_dim = state_dim + latent_dim + step_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Init
        _orthogonal_init(self.net[0], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[2], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[4], gain=0.01)  # small output init

    def forward(
        self, state: torch.Tensor, z_k: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Compute intra-chain value at step k.

        Args:
            state: shape (batch, state_dim).
            z_k: shape (batch, latent_dim).
            k: step index (0 = final latent, K = pure noise).

        Returns:
            value: shape (batch,).
        """
        batch = state.shape[0]
        device = state.device
        k_idx = torch.full((batch,), k, device=device, dtype=torch.long)
        k_embed = self.step_embed(k_idx)
        inp = torch.cat([state, z_k, k_embed], dim=-1)
        return self.net(inp).squeeze(-1)

    def compute_intra_advantages(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        env_advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-step advantages via intra-chain temporal differencing.

        A_{t,k} = V(s, z_{k-1}, k-1) - V(s, z_k, k)

        for k = K, K-1, ..., 1 (matching the policy's step ordering).

        The advantages are normalized to sum to the environment advantage
        via a residual correction, ensuring they're grounded in the actual
        task reward signal.

        Args:
            state: shape (batch, state_dim).
            latents: list [z_K, z_{K-1}, ..., z_0], length K+1.
            env_advantage: shape (batch,), GAE advantage from environment.

        Returns:
            per_step_advantages: shape (batch, K).
        """
        batch = state.shape[0]
        K = self.K

        # Compute V at each step: V_K, V_{K-1}, ..., V_0
        # latents[0] = z_K, latents[1] = z_{K-1}, ..., latents[K] = z_0
        # So latents[i] corresponds to step (K - i)
        values = []
        for i in range(K + 1):
            step_idx = K - i  # latents[0]->step K, latents[K]->step 0
            v = self.forward(state, latents[i], step_idx)
            values.append(v)
        # values[0] = V(s, z_K, K), values[K] = V(s, z_0, 0)

        # Raw intra-chain advantages (matching policy step ordering K->1):
        # Step k (iteration i): A_k = V(s, z_{k-1}, k-1) - V(s, z_k, k)
        # In our indexing: V at z_{k-1} is values[K-k+1], V at z_k is values[K-k]
        # But simpler: step iteration i corresponds to transition latents[i] -> latents[i+1]
        # So A for iteration i = values[i+1] - values[i]  (value of result minus value of input)
        raw_adv = []
        for i in range(K):
            a_i = values[i + 1] - values[i]  # V(z_{k-1}) - V(z_k)
            raw_adv.append(a_i)
        raw_adv = torch.stack(raw_adv, dim=-1)  # (batch, K)

        # Normalize: scale raw advantages so they sum to env_advantage.
        # When raw_sum ≈ 0, fall back to uniform splitting.
        raw_sum = raw_adv.sum(dim=-1, keepdim=True)  # (batch, 1)
        use_uniform = (raw_sum.abs() < 1e-6)  # (batch, 1) bool mask
        # Safe division: use abs + eps to avoid sign issues
        safe_denom = raw_sum.abs().clamp(min=1e-6) * raw_sum.sign()
        # Where raw_sum ≈ 0, set denom to 1 (will be overridden by uniform fallback)
        safe_denom = torch.where(use_uniform, torch.ones_like(safe_denom), safe_denom)
        scale = env_advantage.unsqueeze(-1) / safe_denom
        scale = scale.clamp(-10.0, 10.0)
        per_step_adv = raw_adv * scale
        # Fallback: when raw advantages are near-zero, split uniformly
        uniform_adv = env_advantage.unsqueeze(-1) / K
        per_step_adv = torch.where(use_uniform.expand_as(per_step_adv), uniform_adv, per_step_adv)

        return per_step_adv

    def compute_loss(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        env_advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Train the intra-chain value function.

        Loss: (V(s, z_0, 0) - V(s, z_K, K) - A_t)^2

        The total value change across the chain should predict the
        environment advantage.

        Args:
            state: shape (batch, state_dim).
            latents: list [z_K, ..., z_0], length K+1.
            env_advantage: shape (batch,).

        Returns:
            Scalar loss.
        """
        v_start = self.forward(state, latents[0], self.K)   # V(s, z_K, K)
        v_end = self.forward(state, latents[-1], 0)          # V(s, z_0, 0)
        predicted_advantage = v_end - v_start
        pred_loss = 0.5 * ((predicted_advantage - env_advantage) ** 2).mean()
        # Anchoring: V(s, z_K, K) ≈ 0 (pure noise has no action value)
        anchor_loss = 0.5 * (v_start ** 2).mean()
        return pred_loss + 0.01 * anchor_loss


class StepConditionedValueFunction(nn.Module):
    """V(s, k) -- value at step position k, WITHOUT z_k conditioning.

    Avoids the cold-start problem because the value estimate does not depend
    on near-random latents that are meaningless early in training.  The
    network only sees the environment state and the discrete step index.

    Args:
        state_dim: Observation dimension.
        K: Number of flow steps.
        hidden_dim: Hidden layer width.
        step_embed_dim: Dimension of sinusoidal step embedding.
    """

    def __init__(
        self,
        state_dim: int,
        K: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.step_embed = SinusoidalStepEmbedding(step_embed_dim)

        input_dim = state_dim + step_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        _orthogonal_init(self.net[0], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[2], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[4], gain=0.01)

    def forward(self, state: torch.Tensor, k: int) -> torch.Tensor:
        """Compute value at step *k*.

        Args:
            state: shape (batch, state_dim).
            k: step index (0 = final action, K = pure noise).

        Returns:
            value: shape (batch,).
        """
        batch = state.shape[0]
        k_idx = torch.full((batch,), k, device=state.device, dtype=torch.long)
        k_embed = self.step_embed(k_idx)
        return self.net(torch.cat([state, k_embed], dim=-1)).squeeze(-1)

    def compute_intra_advantages(
        self,
        state: torch.Tensor,
        env_advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Per-step advantages: A_k = V(s, k-1) - V(s, k).  No latents needed.

        Advantages are rescaled so they sum to the environment advantage.

        Args:
            state: shape (batch, state_dim).
            env_advantage: shape (batch,), GAE advantage from environment.

        Returns:
            per_step_advantages: shape (batch, K).
        """
        K = self.K
        # values[0] = V(s, K), values[1] = V(s, K-1), ..., values[K] = V(s, 0)
        values = [self.forward(state, K - i) for i in range(K + 1)]
        raw_adv = torch.stack(
            [values[i + 1] - values[i] for i in range(K)], dim=-1
        )  # (batch, K)

        raw_sum = raw_adv.sum(dim=-1, keepdim=True)
        use_uniform = raw_sum.abs() < 1e-6
        safe_denom = raw_sum.abs().clamp(min=1e-6) * raw_sum.sign()
        safe_denom = torch.where(use_uniform, torch.ones_like(safe_denom), safe_denom)
        scale = (env_advantage.unsqueeze(-1) / safe_denom).clamp(-10.0, 10.0)
        per_step_adv = raw_adv * scale
        uniform_adv = env_advantage.unsqueeze(-1) / K
        return torch.where(use_uniform.expand_as(per_step_adv), uniform_adv, per_step_adv)

    def compute_loss(
        self,
        state: torch.Tensor,
        env_advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Train the step-conditioned value function.

        Loss: (V(s, 0) - V(s, K) - A_t)^2 + anchor on V(s, K).

        Args:
            state: shape (batch, state_dim).
            env_advantage: shape (batch,).

        Returns:
            Scalar loss.
        """
        v_noise = self.forward(state, self.K)
        v_action = self.forward(state, 0)
        pred_loss = 0.5 * ((v_action - v_noise - env_advantage) ** 2).mean()
        anchor_loss = 0.5 * (v_noise ** 2).mean()
        return pred_loss + 0.01 * anchor_loss
