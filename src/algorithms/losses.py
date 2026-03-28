"""PPO loss functions for stochastic flow policies.

Supports four modes of credit assignment:
  1. Holistic: one ratio per trajectory step (standard PPO).
  2. Per-step uniform: factor the chain into K steps with 1/K credit each.
  3. Per-step learned global: replace 1/K with softmax-normalized learned weights.
  4. Per-step state-dependent: weights from a network conditioned on (state, z_k, k).

Modes 2-4 share the same `compute_per_step_ppo_loss` function; the caller
passes in the appropriate weight tensor.
"""

from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Mode 1: Holistic PPO
# ---------------------------------------------------------------------------


def compute_holistic_ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Standard PPO clipped surrogate loss.

    Args:
        log_probs_new: New policy log probs, shape (batch,).
        log_probs_old: Old policy log probs, shape (batch,).
        advantages: Advantage estimates, shape (batch,).
        clip_eps: Clipping epsilon.

    Returns:
        loss: Scalar policy loss (to be minimised).
        info: Dict with clip_fraction, approx_kl, mean_ratio.
    """
    log_ratio = log_probs_new - log_probs_old
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)  # prevent exp overflow
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    # Diagnostics
    with torch.no_grad():
        clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()  # better KL approx
        mean_ratio = ratio.mean()

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
    }
    return loss, info


# ---------------------------------------------------------------------------
# Modes 2/3/4: Per-step PPO
# ---------------------------------------------------------------------------


def compute_per_step_ppo_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: Optional[float] = None,
    clip_eps_low: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-step PPO loss with weighted credit assignment.

    Args:
        step_log_probs_new: shape (batch, K), new per-step log probs.
        step_log_probs_old: shape (batch, K), old per-step log probs.
        advantages: shape (batch,), advantage estimates.
        weights: shape (batch, K), credit weights summing to 1 along dim=-1.
        clip_eps: Clipping epsilon (used as default for both bounds).
        clip_eps_high: Upper clip bound override (ratio clamped to 1 + eps_hi).
            If None, falls back to clip_eps.
        clip_eps_low: Lower clip bound override (ratio clamped to 1 - eps_lo).
            If None, falls back to clip_eps.

    Returns:
        loss: Scalar policy loss.
        info: Dict with clip_fraction, approx_kl, mean_ratio,
              per_step_clip_fractions (K,), per_step_mean_ratios (K,).
    """
    eps_hi = clip_eps_high if clip_eps_high is not None else clip_eps
    eps_lo = clip_eps_low if clip_eps_low is not None else clip_eps

    # Per-step importance ratios: (batch, K)
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)  # prevent exp overflow
    ratios = torch.exp(log_ratios)

    # Weighted advantages: (batch, K)
    weighted_adv = weights * advantages.unsqueeze(-1)  # w_k * A

    surr1 = ratios * weighted_adv
    surr2 = torch.clamp(ratios, 1.0 - eps_lo, 1.0 + eps_hi) * weighted_adv

    # Sum over steps, mean over batch
    loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    # Diagnostics
    with torch.no_grad():
        clipped_high = (ratios - 1.0) > eps_hi
        clipped_low = (1.0 - ratios) > eps_lo
        per_step_clip_fractions = (
            (clipped_high | clipped_low).float().mean(dim=0)
        )  # (K,)
        per_step_mean_ratios = ratios.mean(dim=0)  # (K,)

        clip_fraction = per_step_clip_fractions.mean()
        approx_kl = ((ratios - 1) - log_ratios).mean()  # better KL approx
        mean_ratio = ratios.mean()

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip_fractions,
        "per_step_mean_ratios": per_step_mean_ratios,
    }
    return loss, info


# ---------------------------------------------------------------------------
# Mode 8: Fisher-scaled per-step PPO (natural gradient trust regions)
# ---------------------------------------------------------------------------


def compute_fisher_scaled_ppo_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    sigmas: torch.Tensor,
    K: int,
    base_clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-step PPO loss with Fisher-information-scaled clip epsilon.

    The Fisher information for Gaussian transition k is F_k = diag(1/sigma_k^2).
    The natural gradient scales updates by sigma_k^2, giving steps with higher
    noise a wider trust region (larger clip epsilon) and steps with lower noise
    a tighter trust region.

    The per-step clip epsilon is:
        eps_k = base_clip_eps * (sigma_k^2 / mean(sigma^2))

    This allocates the total KL budget proportionally across steps according
    to their Fisher trace, which is the theoretically optimal allocation.

    With uniform credit weights (1/K per step).

    Args:
        step_log_probs_new: shape (batch, K), new per-step log probs.
        step_log_probs_old: shape (batch, K), old per-step log probs.
        advantages: shape (batch,), advantage estimates.
        sigmas: shape (batch, K) for per-step sigmas, or shape () / (1,) for
            scalar sigma shared across all steps.
        K: number of flow steps.
        base_clip_eps: base clipping epsilon (maps to the mean sigma^2).

    Returns:
        loss: Scalar policy loss.
        info: Dict with clip_fraction, approx_kl, mean_ratio,
              per_step_clip_fractions (K,), per_step_eps (K,).
    """
    # --- Compute per-step clip epsilon scaled by sigma^2 ---
    sigma_sq = sigmas ** 2

    if sigma_sq.dim() <= 1 and sigma_sq.numel() == 1:
        # Scalar sigma: all steps have equal Fisher -> uniform scaling.
        # Scale by sqrt(K) so total KL budget matches holistic.
        # Rationale: K independent per-step clips of eps each give total
        # KL ~ K * eps^2; to match holistic KL ~ eps_h^2, need
        # eps_per_step ~ eps_h (no reduction needed since credit is 1/K).
        eps_per_step = base_clip_eps  # (scalar)
        eps_for_diag = None
    else:
        # sigma_sq: (batch, K) — different sigma per step
        mean_sigma_sq = sigma_sq.mean(dim=-1, keepdim=True)  # (batch, 1)
        eps_per_step = base_clip_eps * sigma_sq / (mean_sigma_sq + 1e-8)  # (batch, K)
        eps_for_diag = eps_per_step.detach().mean(dim=0)  # (K,) for logging

    # --- Per-step importance ratios ---
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)
    ratios = torch.exp(log_ratios)

    # Uniform credit: 1/K per step
    weighted_adv = advantages.unsqueeze(-1) / K  # (batch, K)

    # --- Per-step clipping with Fisher-scaled epsilon ---
    if isinstance(eps_per_step, float) or (isinstance(eps_per_step, torch.Tensor) and eps_per_step.dim() == 0):
        clipped = torch.clamp(ratios, 1.0 - eps_per_step, 1.0 + eps_per_step)
    else:
        # eps_per_step: (batch, K) — per-step and per-sample clipping
        clipped = torch.clamp(ratios, 1.0 - eps_per_step, 1.0 + eps_per_step)

    surr1 = ratios * weighted_adv
    surr2 = clipped * weighted_adv

    # Sum over steps, mean over batch
    loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    # --- Diagnostics ---
    with torch.no_grad():
        if isinstance(eps_per_step, float) or (isinstance(eps_per_step, torch.Tensor) and eps_per_step.dim() == 0):
            eps_val = float(eps_per_step)
            per_step_clip_fractions = (
                (torch.abs(ratios - 1.0) > eps_val).float().mean(dim=0)
            )  # (K,)
            per_step_eps = torch.full(
                (K,), eps_val, device=step_log_probs_new.device
            )
        else:
            per_step_clip_fractions = (
                (torch.abs(ratios - 1.0) > eps_per_step).float().mean(dim=0)
            )
            per_step_eps = eps_for_diag

        clip_fraction = per_step_clip_fractions.mean()
        approx_kl = ((ratios - 1) - log_ratios).mean()
        mean_ratio = ratios.mean()

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip_fractions,
        "per_step_eps": per_step_eps,
    }
    return loss, info


# ---------------------------------------------------------------------------
# Value loss
# ---------------------------------------------------------------------------


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_vf: Optional[float] = None,
) -> torch.Tensor:
    """Mean squared error value loss, optionally with clipping.

    Args:
        values: Predicted values, shape (batch,).
        returns: Target returns, shape (batch,).
        old_values: Old value predictions for clipping, shape (batch,).
        clip_vf: If not None, clip value function updates.

    Returns:
        Scalar value loss.
    """
    if clip_vf is not None and old_values is not None:
        # Clipped value loss (PPO-style)
        values_clipped = old_values + torch.clamp(
            values - old_values, -clip_vf, clip_vf
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        return 0.5 * torch.max(vf_loss1, vf_loss2).mean()
    else:
        return 0.5 * ((values - returns) ** 2).mean()


# ---------------------------------------------------------------------------
# Entropy bonus
# ---------------------------------------------------------------------------


def compute_entropy_bonus(step_log_probs: torch.Tensor) -> torch.Tensor:
    """Approximate entropy of the stochastic flow policy.

    For a K-step stochastic flow, approximate the entropy as the negative
    mean of the sum of per-step log probabilities.

    Args:
        step_log_probs: shape (batch, K).

    Returns:
        Scalar entropy estimate (higher is more stochastic).
    """
    return -step_log_probs.sum(dim=-1).mean()


def compute_weight_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Entropy of credit-assignment weights (encourages uniform distribution).

    Args:
        weights: shape (batch, K), must sum to 1 along dim=-1.

    Returns:
        Scalar mean entropy across the batch. Higher = more uniform.
    """
    # H(w) = -sum(w * log(w)), clamped for numerical safety
    log_w = torch.log(weights.clamp(min=1e-8))
    return -(weights * log_w).sum(dim=-1).mean()


def compute_hierarchical_ppo_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    delta: torch.Tensor,
    K: int,
    clip_eps: float = 0.2,
    clip_eps_high: Optional[float] = None,
    clip_eps_low: Optional[float] = None,
    delta_reg: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hierarchical advantage PPO loss.

    A_{t,k} = A_t / K + delta_k, where delta sums to 0.
    The holistic advantage is preserved; corrections redistribute credit.

    Args:
        step_log_probs_new: shape (batch, K).
        step_log_probs_old: shape (batch, K).
        advantages: shape (batch,), environment GAE advantages.
        delta: shape (batch, K), zero-mean corrections.
        K: number of flow steps.
        clip_eps: PPO clipping epsilon (used as default for both bounds).
        clip_eps_high: Upper clip bound override (ratio clamped to 1 + eps_hi).
            If None, falls back to clip_eps.
        clip_eps_low: Lower clip bound override (ratio clamped to 1 - eps_lo).
            If None, falls back to clip_eps.
        delta_reg: L2 regularization on corrections.

    Returns:
        loss: scalar.
        info: dict with diagnostics.
    """
    eps_hi = clip_eps_high if clip_eps_high is not None else clip_eps
    eps_lo = clip_eps_low if clip_eps_low is not None else clip_eps

    # Per-step advantages: uniform baseline + learned correction
    per_step_adv = advantages.unsqueeze(-1) / K + delta  # (batch, K)

    # Per-step ratios
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)
    ratios = torch.exp(log_ratios)

    surr1 = ratios * per_step_adv
    surr2 = torch.clamp(ratios, 1.0 - eps_lo, 1.0 + eps_hi) * per_step_adv

    policy_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    # L2 regularization: keep corrections small relative to baseline
    reg_loss = delta_reg * (delta ** 2).mean()

    loss = policy_loss + reg_loss

    with torch.no_grad():
        clipped_high = (ratios - 1.0) > eps_hi
        clipped_low = (1.0 - ratios) > eps_lo
        per_step_clip = (clipped_high | clipped_low).float().mean(dim=0)
        clip_fraction = per_step_clip.mean()
        approx_kl = ((ratios - 1) - log_ratios).mean()
        mean_ratio = ratios.mean()
        delta_mag = delta.abs().mean()
        baseline_mag = advantages.abs().mean() / K

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip,
        "delta_magnitude": delta_mag,
        "baseline_magnitude": baseline_mag,
        "delta_ratio": delta_mag / (baseline_mag + 1e-8),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Mode 6: Cumulative-product PPO
# ---------------------------------------------------------------------------


def compute_cumulative_ppo_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    K: int,
    clip_eps: float = 0.2,
    eps_scaling: str = "sqrt",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Cumulative-product PPO loss with adaptive per-prefix clipping.

    Clips R_k = prod_{j=1}^k r_j at each prefix with eps_k = eps * sqrt(k).
    Uses uniform credit (A_t / K per step).
    """
    # Per-step log ratios
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)

    # Cumulative log ratios
    cum_log_ratios = torch.cumsum(log_ratios, dim=-1)  # (batch, K)
    cum_ratios = torch.exp(cum_log_ratios)

    # Step-dependent clipping bounds
    k_indices = torch.arange(1, K + 1, device=step_log_probs_new.device, dtype=torch.float32)
    if eps_scaling == "sqrt":
        eps_k = clip_eps * torch.sqrt(k_indices)
    elif eps_scaling == "linear":
        eps_k = clip_eps * k_indices
    else:
        eps_k = clip_eps * torch.ones_like(k_indices)

    lower = (1.0 - eps_k).unsqueeze(0)  # (1, K)
    upper = (1.0 + eps_k).unsqueeze(0)
    clipped = torch.clamp(cum_ratios, lower, upper)

    # Uniform credit per step
    adv_per_step = advantages.unsqueeze(-1) / K  # (batch, K)

    surr1 = cum_ratios * adv_per_step
    surr2 = clipped * adv_per_step
    loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    with torch.no_grad():
        per_step_clip = ((cum_ratios < lower) | (cum_ratios > upper)).float().mean(dim=0)
        clip_fraction = per_step_clip.mean()
        approx_kl = ((cum_ratios[:, -1] - 1) - cum_log_ratios[:, -1]).mean()
        mean_ratio = cum_ratios[:, -1].mean()

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip,
    }
    return loss, info


# ---------------------------------------------------------------------------
# Mode 7: Hierarchical + Cumulative-product PPO
# ---------------------------------------------------------------------------


def compute_hierarchical_cumulative_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    delta: torch.Tensor,
    K: int,
    clip_eps: float = 0.2,
    delta_reg: float = 0.01,
    eps_scaling: str = "sqrt",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hierarchical advantage with cumulative-product ratio clipping.

    Combines two ideas:
      - Hierarchical advantages: A_k = A/K + delta_k (zero-mean corrections
        redistribute credit from the uniform baseline).
      - Cumulative clipping: clips the running product R_k = prod_{j=1}^k r_j
        with eps_k = eps * sqrt(k), which is less conservative than per-step
        clipping.

    Args:
        step_log_probs_new: shape (batch, K).
        step_log_probs_old: shape (batch, K).
        advantages: shape (batch,), environment GAE advantages.
        delta: shape (batch, K), zero-mean corrections from correction network.
        K: number of flow steps.
        clip_eps: PPO clipping epsilon.
        delta_reg: L2 regularization coefficient on corrections.
        eps_scaling: how clip bounds grow with step index ("sqrt", "linear", "none").

    Returns:
        loss: scalar.
        info: dict with diagnostics.
    """
    # Per-step hierarchical advantages: uniform baseline + learned correction
    per_step_adv = advantages.unsqueeze(-1) / K + delta  # (batch, K)

    # Per-step log ratios -> cumulative log ratios -> cumulative ratios
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)
    cum_log_ratios = torch.cumsum(log_ratios, dim=-1)  # (batch, K)
    cum_ratios = torch.exp(cum_log_ratios)

    # Step-dependent clipping bounds: eps_k = eps * f(k)
    k_indices = torch.arange(1, K + 1, device=step_log_probs_new.device, dtype=torch.float32)
    if eps_scaling == "sqrt":
        eps_k = clip_eps * torch.sqrt(k_indices)
    elif eps_scaling == "linear":
        eps_k = clip_eps * k_indices
    else:
        eps_k = clip_eps * torch.ones_like(k_indices)

    lower = (1.0 - eps_k).unsqueeze(0)  # (1, K)
    upper = (1.0 + eps_k).unsqueeze(0)
    clipped = torch.clamp(cum_ratios, lower, upper)

    # Clipped surrogate with hierarchical per-step advantages
    surr1 = cum_ratios * per_step_adv
    surr2 = clipped * per_step_adv
    policy_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    # L2 regularization on corrections
    reg_loss = delta_reg * (delta ** 2).mean()

    loss = policy_loss + reg_loss

    with torch.no_grad():
        per_step_clip = ((cum_ratios < lower) | (cum_ratios > upper)).float().mean(dim=0)
        clip_fraction = per_step_clip.mean()
        approx_kl = ((cum_ratios[:, -1] - 1) - cum_log_ratios[:, -1]).mean()
        mean_ratio = cum_ratios[:, -1].mean()
        delta_mag = delta.abs().mean()
        baseline_mag = advantages.abs().mean() / K

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip,
        "delta_magnitude": delta_mag,
        "baseline_magnitude": baseline_mag,
        "delta_ratio": delta_mag / (baseline_mag + 1e-8),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Per-step V-MPO (softmax-weighted MLE, no importance ratios)
# ---------------------------------------------------------------------------


def compute_vmpo_loss(
    step_log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    K: int,
    log_etas: torch.Tensor,
    eps_eta: float = 0.01,
    top_frac: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-step V-MPO: softmax-weighted MLE with learned temperatures.

    For each flow step k, filters to the top fraction of the batch by
    advantage, computes softmax weights w_i = softmax(A_i / eta_k), and
    minimises the weighted negative log-likelihood under the current policy.

    No importance ratios are used anywhere.

    Args:
        step_log_probs_new: shape (batch, K), current policy per-step log probs.
        advantages: shape (batch,), GAE advantage estimates.
        K: number of flow steps.
        log_etas: shape (K,), learnable log-temperatures (nn.Parameter).
        eps_eta: KL constraint target per step for the dual temperature loss.
        top_frac: fraction of batch to keep (by advantage).

    Returns:
        policy_loss: Scalar weighted-MLE policy loss (averaged over K steps).
        eta_loss: Scalar dual temperature loss (averaged over K steps).
        info: Dict with diagnostics.
    """
    import math

    batch = step_log_probs_new.shape[0]

    # Filter to top fraction by advantage
    n_keep = max(int(batch * top_frac), 1)
    top_idx = torch.topk(advantages, n_keep).indices
    top_lp = step_log_probs_new[top_idx]  # (n_keep, K)
    top_adv = advantages[top_idx]  # (n_keep,)

    policy_loss = torch.tensor(0.0, device=step_log_probs_new.device)
    eta_loss = torch.tensor(0.0, device=step_log_probs_new.device)

    for k in range(K):
        eta_k = torch.exp(log_etas[k]).clamp(min=1e-4)

        # Softmax weights over filtered batch (detached -- no gradient through weights)
        logits = top_adv / eta_k
        weights = torch.softmax(logits, dim=0).detach()

        # Weighted negative log-likelihood at step k
        step_loss = -(weights * top_lp[:, k]).sum()
        policy_loss = policy_loss + step_loss

        # Temperature dual loss: eta * eps + eta * (logsumexp(A/eta) - log(N))
        step_eta_loss = eta_k * eps_eta + eta_k * (
            torch.logsumexp(top_adv / eta_k, dim=0) - math.log(n_keep)
        )
        eta_loss = eta_loss + step_eta_loss

    policy_loss = policy_loss / K
    eta_loss = eta_loss / K

    with torch.no_grad():
        approx_kl = torch.tensor(0.0)  # no ratios = no KL in traditional sense
        clip_fraction = torch.tensor(0.0)  # no clipping

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": torch.tensor(1.0),
        "etas": [torch.exp(log_etas[k]).item() for k in range(K)],
    }
    return policy_loss, eta_loss, info


# ---------------------------------------------------------------------------
# AWFM: Advantage-Weighted Flow Matching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mode 9: KL Budget Waterfilling PPO
# ---------------------------------------------------------------------------


def compute_waterfill_ppo_loss(
    step_log_probs_new: torch.Tensor,
    step_log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    K: int,
    total_kl_budget: float = 0.02,
    min_budget_frac: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-step PPO with waterfilling KL budget allocation.

    Derives from channel capacity theory: the optimal allocation of a total
    KL budget D across K flow steps is d_k = D * alpha_k^2 / sum_j alpha_j^2,
    where alpha_k is the per-step advantage magnitude (or a proxy thereof).

    On the first PPO epoch (ratios ~ 1.0), per-step KL is near zero, so we
    use the per-step ratio deviation |r_k - 1| as a proxy for alpha_k.
    When that is also uninformative (all near zero), we fall back to uniform
    allocation d_k = D / K.

    The KL budget per step is converted to a clip epsilon via the Gaussian
    approximation: eps_k = sqrt(2 * d_k).

    Args:
        step_log_probs_new: shape (batch, K), new per-step log probs.
        step_log_probs_old: shape (batch, K), old per-step log probs.
        advantages: shape (batch,), advantage estimates.
        K: number of flow steps.
        total_kl_budget: total KL divergence budget D across all steps.
        min_budget_frac: minimum fraction of budget per step (floor).

    Returns:
        loss: Scalar policy loss.
        info: Dict with diagnostics.
    """
    # Per-step importance ratios: (batch, K)
    log_ratios = step_log_probs_new - step_log_probs_old
    log_ratios = torch.clamp(log_ratios, -20.0, 20.0)
    ratios = torch.exp(log_ratios)

    # --- Waterfilling budget allocation (no grad) ---
    with torch.no_grad():
        # Per-step KL: E_batch[ (r_k - 1) - log(r_k) ]  (always >= 0)
        per_step_kl = ((ratios - 1) - log_ratios).mean(dim=0)  # (K,)

        # Use per-step KL as the allocation signal (proxy for alpha_k^2).
        # If KL is too small (first epoch), use |r_k - 1| as fallback.
        alpha_sq = per_step_kl  # (K,)
        total_signal = alpha_sq.sum()

        if total_signal < 1e-10:
            # Fallback: ratio deviation |r_k - 1| averaged across batch
            ratio_dev = (ratios - 1.0).abs().mean(dim=0)  # (K,)
            alpha_sq = ratio_dev ** 2 + 1e-10
            total_signal = alpha_sq.sum()

        if total_signal < 1e-10:
            # Ultimate fallback: uniform allocation
            budgets = torch.full((K,), total_kl_budget / K,
                                 device=step_log_probs_new.device)
        else:
            # Waterfill: d_k = D * alpha_k^2 / sum_j alpha_j^2
            budgets = total_kl_budget * alpha_sq / total_signal  # (K,)

        # Floor: ensure minimum budget per step
        floor = total_kl_budget * min_budget_frac
        budgets = budgets.clamp(min=floor)
        # Renormalize so budgets still sum to D
        budgets = budgets * (total_kl_budget / budgets.sum())

        # Convert KL budget to clip epsilon: eps ~ sqrt(2 * d)
        eps_per_step = torch.sqrt(2.0 * budgets)  # (K,)

    # Uniform credit: 1/K per step
    weighted_adv = advantages.unsqueeze(-1) / K  # (batch, K)

    # Per-step clipping with waterfill-allocated epsilon
    lower = (1.0 - eps_per_step).unsqueeze(0)  # (1, K)
    upper = (1.0 + eps_per_step).unsqueeze(0)  # (1, K)
    clipped = torch.clamp(ratios, lower, upper)

    surr1 = ratios * weighted_adv
    surr2 = clipped * weighted_adv

    # Sum over steps, mean over batch
    loss = -torch.min(surr1, surr2).sum(dim=-1).mean()

    # --- Diagnostics ---
    with torch.no_grad():
        per_step_clip = ((ratios < lower) | (ratios > upper)).float().mean(dim=0)
        clip_fraction = per_step_clip.mean()
        approx_kl = ((ratios - 1) - log_ratios).mean()
        mean_ratio = ratios.mean()

    info = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "per_step_clip_fractions": per_step_clip,
        "per_step_eps": eps_per_step,
        "per_step_budgets": budgets,
    }
    return loss, info


# ---------------------------------------------------------------------------
# AWFM: Advantage-Weighted Flow Matching
# ---------------------------------------------------------------------------


def compute_awfm_loss(
    policy,
    states: torch.Tensor,
    latents,
    noises,
    advantages: torch.Tensor,
    beta: float = 1.0,
    top_frac: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Advantage-Weighted Flow Matching loss.

    Regress the velocity network toward the velocities that produced the
    stored trajectory, weighted by exponentiated advantage.  No importance
    ratios, no clipping -- pure weighted regression.

    Loss per flow step k::

        L_k = w_t * ||v_theta(s, z_k, k) - v_target_k||^2

    where ``w_t = exp(A_t / beta)`` (normalized across the batch) and
    ``v_target_k`` is the velocity that was actually used during rollout,
    reconstructed as ``(z_{k-1} - z_k - sigma * eps) / dt``.

    Args:
        policy: :class:`StochasticFlowPolicy` used to compute current
            velocities and sigmas.
        states: Observation tensor, shape ``(batch, state_dim)``.
        latents: List of ``K+1`` tensors ``[z_K, z_{K-1}, ..., z_0]``,
            each ``(batch, latent_dim)``.
        noises: List of ``K`` tensors ``[eps_K, eps_{K-1}, ..., eps_1]``,
            each ``(batch, latent_dim)``.
        advantages: GAE advantages, shape ``(batch,)``.
        beta: Temperature for the exponential weighting.  Smaller values
            concentrate weight on the highest-advantage samples.
        top_frac: Fraction of the batch to keep (by advantage).  Samples
            below this percentile are discarded before weighting.

    Returns:
        loss: Scalar AWFM loss.
        info: Dict with ``mean_weight``, ``max_weight``, ``effective_batch``
            diagnostic scalars.
    """
    K = policy.K
    dt = 1.0 / K
    batch = states.shape[0]

    # --- Filter to top fraction by advantage ---
    if top_frac < 1.0:
        keep = max(int(batch * top_frac), 1)
        _, top_idx = torch.topk(advantages, keep)
        states = states[top_idx]
        advantages = advantages[top_idx]
        latents = [z[top_idx] for z in latents]
        noises = [e[top_idx] for e in noises]
        batch = keep

    # --- Advantage weights (softmax-style, normalized) ---
    weights = torch.exp(advantages / max(beta, 1e-8))
    weights = weights / (weights.sum() + 1e-8)
    weights = weights.detach()

    s_enc = policy._encode_state(states)

    total_loss = torch.tensor(0.0, device=states.device)
    for i, step in enumerate(range(K, 0, -1)):
        z_k = latents[i]
        z_next = latents[i + 1]
        eps = noises[i]

        k_idx = torch.full((batch,), step, device=states.device, dtype=torch.long)
        k_embed = policy.step_embed(k_idx)

        # Current velocity prediction (through the current network)
        v_pred = policy._velocity(s_enc, z_k, k_embed)

        # Reconstruct the target velocity used during rollout:
        #   z_next = z_k + dt * v + sigma * eps  =>  v = (z_next - z_k - sigma * eps) / dt
        with torch.no_grad():
            sigma = policy._get_sigma(s_enc, z_k, k_embed)
            v_target = (z_next - z_k - sigma * eps) / dt

        # Weighted MSE: w_t * ||v_pred - v_target||^2  (sum over latent dim)
        step_loss = weights * ((v_pred - v_target) ** 2).sum(dim=-1)
        total_loss = total_loss + step_loss.sum()

    loss = total_loss / K

    # Diagnostics
    with torch.no_grad():
        info = {
            "mean_weight": weights.mean(),
            "max_weight": weights.max(),
            "effective_batch": torch.tensor(float(batch), device=states.device),
        }

    return loss, info
