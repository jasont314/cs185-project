#!/usr/bin/env python3
"""
Generate publication-quality plots from stochastic flow policy PPO experiments.

Reads CSV results from results/ directories and produces:
  1. Training curves per environment (method comparison)
  2. Ablation bar charts (final performance)
  3. K-ablation plots

Saves all figures to report/figs/.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results_dir results --output_dir report/figs
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting utilities are self-contained in this script; src.utils.plotting
# provides separate functions for programmatic use.
HAS_PLOTTING_UTILS = False

# ---------------------------------------------------------------------------
# Fallback helpers (used when src.utils.plotting is not yet implemented)
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "holistic": "Holistic",
    "per_step_uniform": "Per-Step Uniform",
    "per_step_learned_global": "Per-Step Learned Global",
    "per_step_state_dependent": "Per-Step State-Dependent",
}

FALLBACK_PALETTE = {
    "holistic": "#1f77b4",
    "per_step_uniform": "#ff7f0e",
    "per_step_learned_global": "#2ca02c",
    "per_step_state_dependent": "#d62728",
}


def _set_default_style():
    """Apply a clean publication style."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })


def _smooth(y, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = y[0] if len(y) > 0 else 0.0
    for val in y:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def _save(fig, path):
    """Save figure and close."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_csv_files(results_dir: str, env: str, method: str) -> list[str]:
    """Return sorted list of CSV paths for a given env/method across seeds."""
    pattern = os.path.join(results_dir, env, method, "seed_*", "progress.csv")
    files = sorted(glob.glob(pattern))
    return files


def load_run(csv_path: str) -> pd.DataFrame | None:
    """Load a single progress.csv into a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  Warning: could not load {csv_path}: {e}")
        return None


def load_method_runs(
    results_dir: str, env: str, method: str
) -> list[pd.DataFrame]:
    """Load all seed runs for a given environment and method."""
    csvs = find_csv_files(results_dir, env, method)
    runs = []
    for csv_path in csvs:
        df = load_run(csv_path)
        if df is not None:
            runs.append(df)
    return runs


def aggregate_runs(
    runs: list[pd.DataFrame],
    x_col: str = "timestep",
    y_col: str = "eval_mean_reward",
    smooth_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Aggregate multiple seed runs into (x, mean, std).

    Interpolates all runs onto a common x-axis grid.
    """
    if not runs:
        return None

    # Determine common x range
    x_min = max(r[x_col].min() for r in runs)
    x_max = min(r[x_col].max() for r in runs)
    if x_min >= x_max:
        return None

    x_grid = np.linspace(x_min, x_max, 200)
    interpolated = []
    for r in runs:
        y_interp = np.interp(x_grid, r[x_col].values, r[y_col].values)
        if smooth_weight > 0:
            y_interp = _smooth(y_interp, weight=smooth_weight)
        interpolated.append(y_interp)

    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)
    return x_grid, mean, std


# ---------------------------------------------------------------------------
# Plot 1: Training curves per environment
# ---------------------------------------------------------------------------

def plot_training_curves(
    results_dir: str,
    output_dir: str,
    envs: list[str],
    methods: list[str],
    smooth_weight: float = 0.6,
):
    """Generate one training-curve figure per environment."""
    print("Generating training curves...")

    palette = COLOR_PALETTE if HAS_PLOTTING_UTILS else FALLBACK_PALETTE

    for env in envs:
        fig, ax = plt.subplots()
        has_data = False

        for method in methods:
            runs = load_method_runs(results_dir, env, method)
            if not runs:
                continue

            agg = aggregate_runs(runs, smooth_weight=smooth_weight)
            if agg is None:
                continue

            x, mean, std = agg
            color = palette.get(method, None)
            label = METHOD_LABELS.get(method, method)

            if HAS_PLOTTING_UTILS:
                plot_mean_std(ax, x, mean, std, color=color, label=label)
            else:
                ax.plot(x, mean, label=label, color=color)
                ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

            has_data = True

        if not has_data:
            print(f"  No data found for {env}, skipping.")
            plt.close(fig)
            continue

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Eval Reward")
        ax.set_title(f"{env} - Training Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, f"training_curves_{env}.pdf")
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 2: Ablation bar charts (final performance)
# ---------------------------------------------------------------------------

def plot_ablation_bars(
    results_dir: str,
    output_dir: str,
    envs: list[str],
    methods: list[str],
    last_n: int = 10,
):
    """
    Bar chart comparing final mean performance across methods for each env.

    Uses the average of the last `last_n` evaluation points per seed,
    then reports mean +/- std across seeds.
    """
    print("Generating ablation bar charts...")

    palette = COLOR_PALETTE if HAS_PLOTTING_UTILS else FALLBACK_PALETTE

    for env in envs:
        method_means = []
        method_stds = []
        method_labels = []

        for method in methods:
            runs = load_method_runs(results_dir, env, method)
            if not runs:
                continue

            # Final performance per seed: average of last `last_n` eval points
            seed_finals = []
            for r in runs:
                if "eval_mean_reward" in r.columns:
                    vals = r["eval_mean_reward"].dropna().values
                    if len(vals) >= last_n:
                        seed_finals.append(vals[-last_n:].mean())
                    elif len(vals) > 0:
                        seed_finals.append(vals.mean())

            if not seed_finals:
                continue

            method_means.append(np.mean(seed_finals))
            method_stds.append(np.std(seed_finals))
            method_labels.append(METHOD_LABELS.get(method, method))

        if not method_means:
            print(f"  No data found for {env}, skipping bar chart.")
            continue

        fig, ax = plt.subplots()
        x_pos = np.arange(len(method_means))
        colors = [
            palette.get(m, "#888888")
            for m in methods
            if METHOD_LABELS.get(m, m) in method_labels
        ]
        # Ensure we have enough colors
        while len(colors) < len(method_means):
            colors.append("#888888")

        ax.bar(x_pos, method_means, yerr=method_stds, color=colors,
               capsize=5, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_labels, rotation=20, ha="right")
        ax.set_ylabel("Final Mean Reward")
        ax.set_title(f"{env} - Method Comparison")
        ax.grid(axis="y", alpha=0.3)

        out_path = os.path.join(output_dir, f"ablation_bars_{env}.pdf")
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3: K-ablation plots
# ---------------------------------------------------------------------------

def plot_k_ablation(
    results_dir: str,
    output_dir: str,
    envs: list[str],
    method: str = "holistic",
    k_values: list[int] | None = None,
    last_n: int = 10,
):
    """
    Plot final performance vs. number of flow steps K.

    Expects results to live under:
        results/{env}_K{k}/{method}/seed_*/progress.csv
    e.g. results/halfcheetah_K2/holistic/seed_42/progress.csv
    """
    print("Generating K-ablation plots...")

    if k_values is None:
        k_values = [1, 2, 4, 8, 16]

    for env in envs:
        k_means = []
        k_stds = []
        k_valid = []

        for k in k_values:
            env_k = f"{env}_K{k}"
            runs = load_method_runs(results_dir, env_k, method)
            if not runs:
                continue

            seed_finals = []
            for r in runs:
                if "eval_mean_reward" in r.columns:
                    vals = r["eval_mean_reward"].dropna().values
                    if len(vals) >= last_n:
                        seed_finals.append(vals[-last_n:].mean())
                    elif len(vals) > 0:
                        seed_finals.append(vals.mean())

            if not seed_finals:
                continue

            k_valid.append(k)
            k_means.append(np.mean(seed_finals))
            k_stds.append(np.std(seed_finals))

        if not k_valid:
            print(f"  No K-ablation data found for {env}, skipping.")
            continue

        fig, ax = plt.subplots()
        k_means = np.array(k_means)
        k_stds = np.array(k_stds)

        ax.errorbar(k_valid, k_means, yerr=k_stds, marker="o",
                     capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel("Number of Flow Steps (K)")
        ax.set_ylabel("Final Mean Reward")
        ax.set_title(f"{env} - K Ablation ({METHOD_LABELS.get(method, method)})")
        ax.set_xticks(k_valid)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, f"k_ablation_{env}.pdf")
        _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for stochastic flow policy PPO experiments."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Root results directory (default: results).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="report/figs",
        help="Directory to save figures (default: report/figs).",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.6,
        help="EMA smoothing weight for training curves (0 = none, default: 0.6).",
    )
    parser.add_argument(
        "--envs", nargs="+",
        default=["halfcheetah", "hopper", "antmaze"],
        help="Environments to plot.",
    )
    args = parser.parse_args()

    # Apply plot style
    if HAS_PLOTTING_UTILS:
        set_style()
    else:
        _set_default_style()

    os.makedirs(args.output_dir, exist_ok=True)

    methods = [
        "holistic",
        "per_step_uniform",
        "per_step_learned_global",
        "per_step_state_dependent",
    ]

    # 1. Training curves
    plot_training_curves(
        args.results_dir, args.output_dir, args.envs, methods,
        smooth_weight=args.smooth,
    )

    # 2. Ablation bar charts
    plot_ablation_bars(
        args.results_dir, args.output_dir, args.envs, methods,
    )

    # 3. K-ablation
    plot_k_ablation(
        args.results_dir, args.output_dir, args.envs,
        method="holistic",
        k_values=[1, 2, 4, 8, 16],
    )

    print("Done.")


if __name__ == "__main__":
    main()
