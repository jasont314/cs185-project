"""Publication-quality plotting utilities for stochastic flow PPO experiments."""

import csv
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import numpy as np

# Use seaborn style if available
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # Fall back to default matplotlib style

# Publication defaults
_FIG_WIDTH = 3.5       # inches, single column
_FIG_HEIGHT = 2.8
_FIG_DPI = 300
_FONT_SIZE = 9
_LEGEND_FONT_SIZE = 7
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

plt.rcParams.update({
    "font.size": _FONT_SIZE,
    "axes.labelsize": _FONT_SIZE,
    "axes.titlesize": _FONT_SIZE + 1,
    "xtick.labelsize": _FONT_SIZE - 1,
    "ytick.labelsize": _FONT_SIZE - 1,
    "legend.fontsize": _LEGEND_FONT_SIZE,
    "figure.dpi": _FIG_DPI,
    "savefig.dpi": _FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving average for smoothing.

    Args:
        values: 1D array of values.
        window: Smoothing window size. If <= 1, no smoothing.

    Returns:
        Smoothed array of the same length.
    """
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    # Pad to keep the same length
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def _read_csv_column(
    csv_path: str, step_col: str, metric_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a single column from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        step_col: Column name for x-axis (e.g. "step").
        metric_col: Column name for the metric.

    Returns:
        Tuple of (steps, values) as numpy arrays.
    """
    steps: list[float] = []
    values: list[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_val = row.get(step_col, "")
            metric_val = row.get(metric_col, "")
            if step_val and metric_val:
                try:
                    steps.append(float(step_val))
                    values.append(float(metric_val))
                except ValueError:
                    continue
    return np.array(steps), np.array(values)


def plot_training_curves(
    csv_paths: Dict[str, str],
    metric: str,
    save_path: str,
    xlabel: str = "Timesteps",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    smooth_window: int = 10,
    step_col: str = "step",
) -> None:
    """Plot training curves from CSV log files.

    Supports multiple runs per label for seed variance shading. In that case,
    pass a dict like ``{"PPO (holistic)": ["run1.csv", "run2.csv", ...]}``.
    For single runs, pass ``{"PPO (holistic)": "run1.csv"}``.

    Args:
        csv_paths: Mapping from label to CSV path(s). Values can be a single
            path string or a list of paths for multiple seeds.
        metric: Column name in the CSV to plot (e.g. "train/mean_reward").
        save_path: File path to save the figure.
        xlabel: X-axis label.
        ylabel: Y-axis label. Defaults to the metric name.
        title: Optional plot title.
        smooth_window: Moving average window for smoothing.
        step_col: Column name used as x-axis.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    if ylabel is None:
        ylabel = metric

    for i, (label, paths) in enumerate(csv_paths.items()):
        color = _COLORS[i % len(_COLORS)]

        if isinstance(paths, str):
            paths = [paths]

        all_values: list[np.ndarray] = []
        common_steps: Optional[np.ndarray] = None

        for p in paths:
            steps, values = _read_csv_column(p, step_col, metric)
            if len(steps) == 0:
                continue
            smoothed = _smooth(values, smooth_window)
            all_values.append(smoothed)
            if common_steps is None or len(steps) < len(common_steps):
                common_steps = steps

        if not all_values or common_steps is None:
            continue

        # Truncate to shortest run
        min_len = min(len(v) for v in all_values)
        common_steps = common_steps[:min_len]
        truncated = np.array([v[:min_len] for v in all_values])

        mean = truncated.mean(axis=0)
        ax.plot(common_steps, mean, label=label, color=color, linewidth=1.2)

        if truncated.shape[0] > 1:
            std = truncated.std(axis=0)
            ax.fill_between(
                common_steps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ablation_bar(
    results: Dict[str, Tuple[float, float]],
    save_path: str,
    ylabel: str = "Return",
    title: Optional[str] = None,
) -> None:
    """Bar chart comparing methods with error bars.

    Args:
        results: Mapping from label to (mean, std).
        save_path: File path to save the figure.
        ylabel: Y-axis label.
        title: Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    labels = list(results.keys())
    means = [results[l][0] for l in labels]
    stds = [results[l][1] for l in labels]
    x = np.arange(len(labels))

    bars = ax.bar(
        x, means, yerr=stds,
        capsize=3, color=_COLORS[:len(labels)],
        edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 0.8},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_weight_profile(
    weights: np.ndarray,
    save_path: str,
    step_labels: Optional[List[str]] = None,
) -> None:
    """Bar chart of learned per-step credit weights.

    Args:
        weights: 1D array of shape (K,).
        save_path: File path to save the figure.
        step_labels: Optional labels for each step. Defaults to "k=1", "k=2", ...
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    K = len(weights)
    x = np.arange(K)
    if step_labels is None:
        step_labels = [f"k={i + 1}" for i in range(K)]

    ax.bar(x, weights, color=_COLORS[0], edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0 / K, color="gray", linestyle="--", linewidth=0.8, label=f"Uniform (1/{K})")
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Weight")
    ax.set_title("Per-step credit weights")
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_weight_heatmap(
    weights_over_time: np.ndarray,
    save_path: str,
    checkpoint_labels: Optional[List[str]] = None,
) -> None:
    """Heatmap showing how per-step weights evolve over training.

    Args:
        weights_over_time: 2D array of shape (num_checkpoints, K).
        save_path: File path to save the figure.
        checkpoint_labels: Optional labels for the y-axis (checkpoints).
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    num_ckpts, K = weights_over_time.shape

    im = ax.imshow(
        weights_over_time.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Weight")

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Denoising step")
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"k={i + 1}" for i in range(K)])

    if checkpoint_labels is not None:
        ax.set_xticks(np.arange(num_ckpts))
        ax.set_xticklabels(checkpoint_labels, rotation=45, ha="right")
    else:
        # Auto tick a reasonable number of checkpoints
        n_ticks = min(num_ckpts, 10)
        tick_positions = np.linspace(0, num_ckpts - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(t) for t in tick_positions])

    ax.set_title("Weight evolution over training")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_per_step_clip_fractions(
    clip_fractions: Dict[str, np.ndarray],
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Grouped bar chart of per-step clip fractions across methods.

    Args:
        clip_fractions: Mapping from label to array of shape (K,).
        save_path: File path to save the figure.
        title: Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    labels = list(clip_fractions.keys())
    n_methods = len(labels)

    # Determine K from first entry
    K = len(next(iter(clip_fractions.values())))
    x = np.arange(K)
    bar_width = 0.8 / n_methods

    for i, label in enumerate(labels):
        fracs = clip_fractions[label]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, fracs,
            width=bar_width, label=label,
            color=_COLORS[i % len(_COLORS)],
            edgecolor="black", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={i + 1}" for i in range(K)])
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Clip fraction")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_k_ablation(
    results: Dict[int, Dict[str, Tuple[float, float]]],
    save_path: str,
    ylabel: str = "Return",
    title: Optional[str] = None,
) -> None:
    """Grouped bar chart for K ablation across methods.

    Args:
        results: Nested dict ``{K_value: {method_name: (mean, std)}}``.
        save_path: File path to save the figure.
        ylabel: Y-axis label.
        title: Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH * 1.3, _FIG_HEIGHT))

    k_values = sorted(results.keys())
    # Collect all method names across all K values
    all_methods: list[str] = []
    for k_val in k_values:
        for m in results[k_val]:
            if m not in all_methods:
                all_methods.append(m)

    n_methods = len(all_methods)
    n_groups = len(k_values)
    x = np.arange(n_groups)
    bar_width = 0.8 / n_methods

    for i, method in enumerate(all_methods):
        means = []
        stds = []
        for k_val in k_values:
            if method in results[k_val]:
                m, s = results[k_val][method]
                means.append(m)
                stds.append(s)
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means,
            width=bar_width, yerr=stds,
            capsize=2, label=method,
            color=_COLORS[i % len(_COLORS)],
            edgecolor="black", linewidth=0.3,
            error_kw={"linewidth": 0.6},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_xlabel("Number of denoising steps (K)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
