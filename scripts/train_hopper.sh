#!/bin/bash
# Train all methods on Hopper

SEEDS="42 123 456"
CONFIG=configs/hopper.yaml

for SEED in $SEEDS; do
  # Holistic PPO
  python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --logging.log_dir results/hopper/holistic/seed_${SEED}

  # Per-step uniform
  python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --logging.log_dir results/hopper/per_step_uniform/seed_${SEED}

  # Per-step learned global
  python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --logging.log_dir results/hopper/per_step_learned_global/seed_${SEED}
done
