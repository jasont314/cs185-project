"""Weighting networks for per-step credit assignment in stochastic flow PPO."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stochastic_flow_policy import SinusoidalStepEmbedding


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class UniformWeights(nn.Module):
    """Return uniform 1/K weights for every step.

    This is a parameter-free baseline.
    """

    def forward(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
        **kwargs,
    ) -> torch.Tensor:
        """Compute uniform weights.

        Args:
            state: Observation tensor, shape (batch, state_dim). Unused.
            latents: List of latent tensors [z_K, ..., z_0]. Unused except for batch/device info.
            K: Number of denoising steps.

        Returns:
            weights: shape (batch, K), each entry = 1/K.
        """
        batch = latents[0].shape[0]
        device = latents[0].device
        return torch.full((batch, K), 1.0 / K, device=device)


class DenoisingDiscountWeights(nn.Module):
    """DPPO-style denoising discount: w_k = gamma^(K-1-k) / Z.

    Downweights noisier (earlier) steps, giving more credit to steps
    closer to the final action. From Ren et al. (ICLR 2025).

    Args:
        gamma_denoise: Discount factor per denoising step.
    """

    def __init__(self, gamma_denoise: float = 0.95) -> None:
        super().__init__()
        self.gamma_denoise = gamma_denoise

    def forward(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
        **kwargs,
    ) -> torch.Tensor:
        batch = latents[0].shape[0]
        device = latents[0].device
        # Step iteration i=0 is step K (noisiest), i=K-1 is step 1 (cleanest)
        # w_i = gamma^(K-1-i), so noisier steps get lower weight
        exponents = torch.arange(K - 1, -1, -1, device=device, dtype=torch.float32)
        raw_w = self.gamma_denoise ** exponents  # (K,)
        w = raw_w / raw_w.sum()  # normalize to sum to 1
        return w.unsqueeze(0).expand(batch, -1)


class LearnedGlobalWeights(nn.Module):
    """Learnable global weight vector, softmax-normalized.

    One set of weights shared across all states.

    Args:
        K: Number of denoising steps.
    """

    def __init__(self, K: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(K))

    def forward(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
    ) -> torch.Tensor:
        """Compute learned global weights.

        Args:
            state: shape (batch, state_dim). Unused.
            latents: list of latent tensors. Used for batch size.
            K: Number of steps (should match self.alpha.shape[0]).

        Returns:
            weights: shape (batch, K), summing to 1.
        """
        batch = latents[0].shape[0]
        w = F.softmax(self.alpha, dim=0)  # (K,)
        return w.unsqueeze(0).expand(batch, -1)  # (batch, K)


class StateDependentWeights(nn.Module):
    """State-dependent weighting via an MLP.

    Takes (state_enc, z_k, step_embedding) at each step and produces
    a scalar logit. The K logits are then softmax-normalized.

    Args:
        latent_dim: Dimension of the state encoding and z_k.
        hidden_dim: Hidden layer width.
        step_embed_dim: Dimension of the step embedding.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        step_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.step_embed = SinusoidalStepEmbedding(step_embed_dim)
        input_dim = latent_dim + latent_dim + step_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        _orthogonal_init(self.net[0], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[2], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[4], gain=0.01)

    def forward(
        self,
        state_enc: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
    ) -> torch.Tensor:
        """Compute state-dependent weights.

        Args:
            state_enc: Encoded state from the policy's state encoder,
                shape (batch, latent_dim).
            latents: List [z_K, z_{K-1}, ..., z_0] of length K+1.
            K: Number of denoising steps.

        Returns:
            weights: shape (batch, K), summing to 1 per sample.
        """
        batch = state_enc.shape[0]
        device = state_enc.device

        logits: list[torch.Tensor] = []
        for i, step in enumerate(range(K, 0, -1)):
            z_k = latents[i]  # z at step k
            k_idx = torch.full((batch,), step, device=device, dtype=torch.long)
            k_embed = self.step_embed(k_idx)
            inp = torch.cat([state_enc, z_k, k_embed], dim=-1)
            logit = self.net(inp).squeeze(-1)  # (batch,)
            logits.append(logit)

        logits_tensor = torch.stack(logits, dim=-1)  # (batch, K)
        return F.softmax(logits_tensor, dim=-1)


class KLInverseWeights(nn.Module):
    """Weight per-step advantages by inverse KL divergence.

    w_k = softmax(-beta * KL_k) where KL_k is per-step approx KL.
    Steps where policy changed less get more credit.

    Args:
        beta: Inverse temperature. Higher = more peaked on low-KL steps.
        learn_beta: If True, beta is learnable.
    """

    def __init__(self, beta: float = 5.0, learn_beta: bool = False) -> None:
        super().__init__()
        if learn_beta:
            self._log_beta = nn.Parameter(torch.tensor(float(beta)).log())
        else:
            self.register_buffer("_log_beta", torch.tensor(float(beta)).log())

    @property
    def beta(self) -> torch.Tensor:
        return self._log_beta.exp().clamp(min=0.1)

    def forward(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
        step_log_probs_new: Optional[torch.Tensor] = None,
        step_log_probs_old: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL-inverse weights.

        Requires step_log_probs_new and step_log_probs_old, each of
        shape (batch, K).

        Args:
            state: Observation tensor, shape (batch, state_dim). Unused.
            latents: List of latent tensors. Unused.
            K: Number of denoising steps.
            step_log_probs_new: Per-step log probs under current policy, (batch, K).
            step_log_probs_old: Per-step log probs under old policy, (batch, K).

        Returns:
            weights: shape (batch, K), summing to 1 per sample. Detached.
        """
        assert step_log_probs_new is not None and step_log_probs_old is not None, (
            "KLInverseWeights requires step_log_probs_new and step_log_probs_old"
        )
        log_ratios = torch.clamp(step_log_probs_new - step_log_probs_old, -20, 20)
        ratios = torch.exp(log_ratios)
        per_step_kl = (ratios - 1) - log_ratios  # (batch, K), >= 0
        weights = F.softmax(-self.beta * per_step_kl, dim=-1)
        return weights.detach()  # detach: credit signal only


class HierarchicalCorrectionNetwork(nn.Module):
    """Learned zero-mean corrections for hierarchical advantage estimation.

    Per-step advantage: A_{t,k} = A_t / K + delta_k(s, z_k, k)
    where sum_k delta_k = 0 by construction (mean-subtraction).

    The holistic advantage is preserved as a guaranteed floor — corrections
    can only redistribute credit, not change the total.

    Args:
        state_dim: Observation dimension.
        latent_dim: Latent z dimension.
        hidden_dim: Hidden layer width.
        step_embed_dim: Step embedding dimension.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        step_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.step_embed = SinusoidalStepEmbedding(step_embed_dim)
        input_dim = state_dim + latent_dim + step_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Small output init so corrections start near zero (≈ uniform baseline)
        _orthogonal_init(self.net[0], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[2], gain=2.0 ** 0.5)
        _orthogonal_init(self.net[4], gain=0.01)

    def forward(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        K: int,
    ) -> torch.Tensor:
        """Compute zero-mean corrections delta_k.

        Args:
            state: shape (batch, state_dim).
            latents: list [z_K, z_{K-1}, ..., z_0], length K+1.
            K: Number of denoising steps.

        Returns:
            delta: shape (batch, K), summing to 0 along dim=-1.
        """
        batch = state.shape[0]
        device = state.device

        raw_deltas: list[torch.Tensor] = []
        for i, step in enumerate(range(K, 0, -1)):
            z_k = latents[i]
            k_idx = torch.full((batch,), step, device=device, dtype=torch.long)
            k_embed = self.step_embed(k_idx)
            inp = torch.cat([state, z_k, k_embed], dim=-1)
            raw_deltas.append(self.net(inp).squeeze(-1))  # (batch,)

        raw = torch.stack(raw_deltas, dim=-1)  # (batch, K)
        # Enforce zero-mean: corrections sum to 0
        delta = raw - raw.mean(dim=-1, keepdim=True)
        return delta
