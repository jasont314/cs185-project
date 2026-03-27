#!/bin/bash
# Train all methods on AntMaze (including state-dependent variant)

SEEDS="42 123 456"
CONFIG=configs/antmaze.yaml

for SEED in $SEEDS; do
  # Holistic PPO
  python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --logging.log_dir results/antmaze/holistic/seed_${SEED}

  # Per-step uniform
  python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --logging.log_dir results/antmaze/per_step_uniform/seed_${SEED}

  # Per-step learned global
  python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --logging.log_dir results/antmaze/per_step_learned_global/seed_${SEED}

  # Per-step state dependent
  python -m src.training.trainer --config $CONFIG --method.mode per_step_state_dependent --seed $SEED --logging.log_dir results/antmaze/per_step_state_dependent/seed_${SEED}
done
