#!/bin/bash
# Train all methods on HalfCheetah

SEEDS="42 123 456"
CONFIG=configs/halfcheetah.yaml

for SEED in $SEEDS; do
  # Holistic PPO
  python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --logging.log_dir results/halfcheetah/holistic/seed_${SEED}

  # Per-step uniform
  python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --logging.log_dir results/halfcheetah/per_step_uniform/seed_${SEED}

  # Per-step learned global
  python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --logging.log_dir results/halfcheetah/per_step_learned_global/seed_${SEED}
done
