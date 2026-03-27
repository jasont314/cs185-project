"""Stochastic flow policy with K internal denoising steps."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.distributions import gaussian_log_prob, sample_gaussian, tanh_squash_correction


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to a linear layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    small_output: bool = False,
) -> nn.Sequential:
    """Build a 2-hidden-layer MLP with orthogonal init.

    Args:
        input_dim: Input feature size.
        hidden_dim: Hidden layer size.
        output_dim: Output feature size.
        small_output: If True, use gain=0.01 for output layer init.
    """
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    # Init hidden layers
    _orthogonal_init(net[0], gain=2.0 ** 0.5)
    _orthogonal_init(net[2], gain=2.0 ** 0.5)
    # Init output layer
    _orthogonal_init(net[4], gain=0.01 if small_output else 1.0)
    return net


class SinusoidalStepEmbedding(nn.Module):
    """Sinusoidal positional embedding for the step index k."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Embed step indices.

        Args:
            k: Integer step indices, shape (batch,) or scalar.

        Returns:
            Embeddings, shape (batch, embed_dim).
        """
        k = k.float()
        if k.dim() == 0:
            k = k.unsqueeze(0)
        half = self.embed_dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=k.device, dtype=torch.float32) / half
        )
        # k: (batch,), freq: (half,)  ->  args: (batch, half)
        args = k.unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, embed_dim)
        return emb


class StochasticFlowPolicy(nn.Module):
    """Stochastic flow policy that generates actions via a K-step denoising chain.

    The generative process:
        z_K ~ N(0, I)
        z_{k-1} = z_k + dt * v_theta(s_enc, z_k, k) + sigma(s_enc, z_k, k) * eps_k
        a = tanh(z_0)

    where dt = 1/K, v_theta is a learned velocity network, and sigma is either
    a fixed scalar or a learned network.

    Args:
        state_dim: Dimension of the observation/state.
        action_dim: Dimension of the action space.
        hidden_dim: Hidden layer width for all MLPs.
        latent_dim: Dimension of the latent z space.
        K: Number of denoising steps.
        sigma_init: Initial value of the noise scale.
        learn_sigma: If True, make the scalar sigma a learnable parameter.
        sigma_network: If True, use a state/step-dependent sigma network
            instead of a scalar. Overrides learn_sigma.
        step_embed_dim: Dimension of the step embedding.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = None,
        K: int = 4,
        sigma_init: float = 1.0,
        learn_sigma: bool = False,
        sigma_network: bool = False,
        step_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        if latent_dim is None:
            latent_dim = action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.K = K
        self.step_embed_dim = step_embed_dim
        self.sigma_network_flag = sigma_network

        # --- State encoder ---
        self.state_encoder = _build_mlp(state_dim, hidden_dim, latent_dim)

        # --- Step embedding ---
        self.step_embed = SinusoidalStepEmbedding(step_embed_dim)

        # --- Velocity / drift network ---
        vel_input_dim = latent_dim + latent_dim + step_embed_dim
        self.velocity_net = _build_mlp(vel_input_dim, hidden_dim, latent_dim, small_output=True)

        # --- Sigma ---
        if sigma_network:
            sigma_input_dim = latent_dim + latent_dim + step_embed_dim
            self.sigma_net = _build_mlp(sigma_input_dim, hidden_dim, latent_dim)
            # We'll use softplus on the output to ensure positivity
        else:
            if learn_sigma:
                self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
            else:
                self.register_buffer("log_sigma", torch.tensor(math.log(sigma_init)))

        # Note: since latent_dim = action_dim by default, z_0 is used directly
        # as the pre-tanh action. No separate action_head is needed — this avoids
        # a bug where the tanh Jacobian correction drifts during PPO updates
        # because action_head params change but stored z_0 values don't.
        assert latent_dim == action_dim, (
            f"latent_dim ({latent_dim}) must equal action_dim ({action_dim}) "
            "so z_0 can be used directly as pre-tanh action"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into latent representation.

        Args:
            state: shape (batch, state_dim).

        Returns:
            s_enc: shape (batch, latent_dim).
        """
        return self.state_encoder(state)

    def _get_sigma(
        self,
        s_enc: torch.Tensor,
        z_k: torch.Tensor,
        k_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Return noise scale sigma for a given step.

        Args:
            s_enc: Encoded state, shape (batch, latent_dim).
            z_k: Current latent, shape (batch, latent_dim).
            k_embed: Step embedding, shape (batch, step_embed_dim).

        Returns:
            sigma: shape (batch, latent_dim) if sigma_network, else broadcastable scalar.
        """
        if self.sigma_network_flag:
            inp = torch.cat([s_enc, z_k, k_embed], dim=-1)
            return F.softplus(self.sigma_net(inp)) + 1e-6
        else:
            return torch.exp(self.log_sigma)

    def _velocity(
        self,
        s_enc: torch.Tensor,
        z_k: torch.Tensor,
        k_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity/drift v_theta.

        Args:
            s_enc: shape (batch, latent_dim).
            z_k: shape (batch, latent_dim).
            k_embed: shape (batch, step_embed_dim).

        Returns:
            v: shape (batch, latent_dim).
        """
        inp = torch.cat([s_enc, z_k, k_embed], dim=-1)
        return self.velocity_net(inp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_chain(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the full K-step denoising chain and return all intermediates.

        Args:
            state: Observation, shape (batch, state_dim).

        Returns:
            Dictionary with keys:
                action: shape (batch, action_dim), tanh-squashed.
                latents: list of tensors [z_K, z_{K-1}, ..., z_0], each (batch, latent_dim).
                noises: list of K noise tensors [eps_K, eps_{K-1}, ..., eps_1],
                    each (batch, latent_dim). eps at index i corresponds to
                    the transition from latents[i] to latents[i+1].
                per_step_log_probs: shape (batch, K).
                holistic_log_prob: shape (batch,).
        """
        batch = state.shape[0]
        device = state.device
        dt = 1.0 / self.K

        s_enc = self._encode_state(state)

        # z_K ~ N(0, I)
        z = torch.randn(batch, self.latent_dim, device=device)

        latents: List[torch.Tensor] = [z]
        noises: List[torch.Tensor] = []
        per_step_lps: List[torch.Tensor] = []

        for step in range(self.K, 0, -1):
            # step goes K, K-1, ..., 1
            k_idx = torch.full((batch,), step, device=device, dtype=torch.long)
            k_embed = self.step_embed(k_idx)

            v = self._velocity(s_enc, z, k_embed)
            sigma_raw = self._get_sigma(s_enc, z, k_embed)
            # SDE-consistent noise: scale by sqrt(dt) so effective diffusion
            # is K-independent. Without this, total noise scales as sigma*sqrt(K).
            sigma = sigma_raw * (dt ** 0.5)

            # Transition mean
            mean = z + dt * v

            # Sample z_{k-1}
            z_next, eps = sample_gaussian(mean, sigma)

            # Per-step log prob: log N(z_next; mean, sigma^2 I)
            lp = gaussian_log_prob(mean, sigma, z_next)

            latents.append(z_next)
            noises.append(eps)
            per_step_lps.append(lp)

            z = z_next

        # z_0 is the last element of latents; use directly as pre-tanh action
        z_0 = latents[-1]
        action = torch.tanh(z_0)

        per_step_log_probs = torch.stack(per_step_lps, dim=-1)  # (batch, K)

        # Holistic log prob = sum of per-step + tanh correction
        tanh_correction = tanh_squash_correction(z_0)  # (batch,)
        holistic_log_prob = per_step_log_probs.sum(dim=-1) - tanh_correction  # (batch,)

        return {
            "action": action,
            "latents": latents,       # [z_K, z_{K-1}, ..., z_0]
            "noises": noises,         # [eps_K, eps_{K-1}, ..., eps_1]
            "per_step_log_probs": per_step_log_probs,
            "holistic_log_prob": holistic_log_prob,
        }

    def compute_per_step_log_probs(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        noises: List[torch.Tensor],
    ) -> torch.Tensor:
        """Recompute per-step log probabilities from stored latents/noises.

        This is used during PPO updates to evaluate the current policy's
        probability of previously collected trajectories.

        Args:
            state: shape (batch, state_dim).
            latents: list [z_K, z_{K-1}, ..., z_0], length K+1.
            noises: list [eps_K, ..., eps_1], length K (unused, kept for API symmetry).

        Returns:
            per_step_log_probs: shape (batch, K).
        """
        s_enc = self._encode_state(state)
        dt = 1.0 / self.K
        batch = state.shape[0]
        device = state.device

        lps: List[torch.Tensor] = []
        for i, step in enumerate(range(self.K, 0, -1)):
            z_k = latents[i]
            z_next = latents[i + 1]

            k_idx = torch.full((batch,), step, device=device, dtype=torch.long)
            k_embed = self.step_embed(k_idx)

            v = self._velocity(s_enc, z_k, k_embed)
            sigma_raw = self._get_sigma(s_enc, z_k, k_embed)
            sigma = sigma_raw * (dt ** 0.5)  # SDE-consistent scaling

            mean = z_k + dt * v
            lp = gaussian_log_prob(mean, sigma, z_next)
            lps.append(lp)

        return torch.stack(lps, dim=-1)  # (batch, K)

    def compute_holistic_log_prob(
        self,
        state: torch.Tensor,
        latents: List[torch.Tensor],
        noises: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the full log probability of the action under the current policy.

        Args:
            state: shape (batch, state_dim).
            latents: list [z_K, ..., z_0], length K+1.
            noises: list [eps_K, ..., eps_1], length K.

        Returns:
            holistic_log_prob: shape (batch,).
        """
        per_step = self.compute_per_step_log_probs(state, latents, noises)
        z_0 = latents[-1]
        tanh_correction = tanh_squash_correction(z_0)
        return per_step.sum(dim=-1) - tanh_correction

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate an action (inference mode, no intermediates).

        Args:
            state: shape (batch, state_dim) or (state_dim,).

        Returns:
            action: shape (batch, action_dim), tanh-squashed.
        """
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        result = self.sample_chain(state)
        action = result["action"]
        if squeeze:
            action = action.squeeze(0)
        return action

    def get_action_and_info(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate an action along with auxiliary information for rollout collection.

        Args:
            state: shape (batch, state_dim).

        Returns:
            action: shape (batch, action_dim).
            info: dict with keys:
                latents, noises, per_step_log_probs, holistic_log_prob
        """
        result = self.sample_chain(state)
        action = result["action"]
        info = {
            "latents": result["latents"],
            "noises": result["noises"],
            "per_step_log_probs": result["per_step_log_probs"],
            "holistic_log_prob": result["holistic_log_prob"],
        }
        return action, info
