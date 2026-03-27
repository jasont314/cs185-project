# Stochastic Flow Policy PPO: Holistic vs Per-Step Credit Assignment

## Overview

This project investigates how credit assignment granularity affects PPO training of stochastic flow policies. A stochastic flow policy generates actions through K iterative denoising steps. We compare four modes of assigning credit across those internal steps:

1. **Holistic** -- treat the entire K-step flow as a single action and apply standard PPO.
2. **Per-step uniform** -- decompose the policy gradient across steps with equal weighting.
3. **Per-step learned global** -- learn a shared weight vector over the K steps.
4. **Per-step state-dependent** -- learn a network that outputs per-step weights conditioned on the current state.

Experiments run on HalfCheetah-v5, Hopper-v5, and AntMaze, comparing sample efficiency and final return across modes.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── configs/
│   ├── halfcheetah.yaml
│   ├── hopper.yaml
│   └── antmaze.yaml
├── scripts/
│   ├── train_halfcheetah.sh
│   ├── train_hopper.sh
│   ├── train_antmaze.sh
│   └── plot_results.py
├── src/
│   ├── algorithms/
│   │   ├── advantages.py        # GAE computation
│   │   ├── buffers.py           # Rollout buffer
│   │   ├── losses.py            # PPO loss functions
│   │   ├── ppo_holistic.py      # Holistic PPO agent
│   │   └── ppo_per_step.py      # Per-step PPO agent
│   ├── envs/
│   │   └── __init__.py          # Environment factory
│   ├── models/
│   │   ├── stochastic_flow_policy.py  # K-step flow policy
│   │   ├── value_function.py          # Value network
│   │   └── weighting_network.py       # Learned step weights
│   ├── training/
│   │   ├── evaluate.py          # Evaluation loop
│   │   ├── logger.py            # CSV / TensorBoard logging
│   │   └── trainer.py           # Trainer class + CLI entry point
│   └── utils/
│       ├── distributions.py
│       ├── math_utils.py
│       └── plotting.py
├── report/                      # LaTeX report
├── results/                     # Training logs and checkpoints
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Train holistic PPO on HalfCheetah
python -m src.training.trainer --config configs/halfcheetah.yaml \
    --method.mode holistic \
    --logging.log_dir results/halfcheetah/holistic/seed_42

# Train per-step uniform PPO
python -m src.training.trainer --config configs/halfcheetah.yaml \
    --method.mode per_step_uniform \
    --logging.log_dir results/halfcheetah/per_step_uniform/seed_42

# Train per-step learned global weights
python -m src.training.trainer --config configs/halfcheetah.yaml \
    --method.mode per_step_learned_global \
    --logging.log_dir results/halfcheetah/per_step_learned_global/seed_42

# Train per-step state-dependent weights
python -m src.training.trainer --config configs/halfcheetah.yaml \
    --method.mode per_step_state_dependent \
    --logging.log_dir results/halfcheetah/per_step_state_dependent/seed_42
```

## Run All Experiments

```bash
bash scripts/train_halfcheetah.sh
bash scripts/train_hopper.sh
bash scripts/train_antmaze.sh
```

## Generate Plots

```bash
python scripts/plot_results.py
```

## Method

Given state s, the stochastic flow policy produces an action through K denoising steps:

```
z_0 ~ N(0, sigma^2 I)
z_{k+1} = z_k + f_theta(z_k, s, k/K)     for k = 0, ..., K-1
a = tanh(z_K)
```

The four credit-assignment modes differ in how the PPO objective is decomposed:

- **Holistic**: L = E[ min(r(theta) A, clip(r(theta), 1-eps, 1+eps) A) ] where r(theta) is the likelihood ratio of the full trajectory z_0 -> z_K.
- **Per-step uniform**: L = (1/K) sum_k L_k, where L_k uses the per-step likelihood ratio at step k.
- **Per-step learned global**: L = sum_k w_k L_k, where w = softmax(alpha) and alpha is a learnable K-dim vector.
- **Per-step state-dependent**: L = sum_k w_k(s) L_k, where w(s) = softmax(g_phi(s)) and g_phi is a small network.

## Configuration

Key parameters (set in YAML or via CLI overrides):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method.mode` | Credit assignment mode | `holistic` |
| `policy.K` | Number of internal flow steps | `4` |
| `policy.sigma_init` | Initial noise scale for z_0 | `0.1` |
| `policy.learn_sigma` | Whether sigma is learnable | `false` |
| `policy.hidden_dim` | Hidden layer width | `256` |
| `policy.latent_dim` | Latent space dimension | `64` |
| `ppo.clip_eps` | PPO clipping epsilon | `0.2` |
| `ppo.lr` | Learning rate | `3e-4` |
| `ppo.num_rollout_steps` | Steps per rollout | `2048` |
| `total_timesteps` | Total training steps | `1000000` |
| `seed` | Random seed | `0` |

## Ablations

Run a sweep over the number of flow steps K:

```bash
for K in 2 4 8; do
  python -m src.training.trainer --config configs/halfcheetah.yaml \
      --policy.K $K \
      --logging.log_dir results/halfcheetah_K${K}/holistic/seed_42
done
```

Run a sweep over noise scale:

```bash
for sigma in 0.01 0.1 0.5; do
  python -m src.training.trainer --config configs/halfcheetah.yaml \
      --policy.sigma_init $sigma \
      --logging.log_dir results/halfcheetah_sigma${sigma}/holistic/seed_42
done
```

## Report

```bash
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
