#!/bin/bash
# Run all method variants on HalfCheetah for the main comparison.
# Runs 3 methods in parallel per seed (fits in 16GB RAM), then next seed.
#
# Usage:
#   bash scripts/run_all.sh

set -e

CONFIG=configs/halfcheetah.yaml
SEEDS="42 123 456"
METHODS="holistic per_step_uniform per_step_learned_global"

echo "=== Stochastic Flow Policy PPO — HalfCheetah Main Comparison ==="
echo "Methods: $METHODS"
echo "Seeds:   $SEEDS"
echo "Running 3 methods in parallel per seed (~4.5GB RAM)"
echo ""

for SEED in $SEEDS; do
  echo "--- Seed ${SEED} ---"
  PIDS=()

  for METHOD in $METHODS; do
    LOG_DIR="results/halfcheetah/${METHOD}/seed_${SEED}"
    mkdir -p "${LOG_DIR}"

    # Skip if already completed
    if [[ -f "${LOG_DIR}/checkpoints/final.pt" ]]; then
      echo "  [SKIP] ${METHOD} — already completed"
      continue
    fi

    echo "  [START] ${METHOD}"
    python -m src.training.trainer \
      --config ${CONFIG} \
      --method.mode ${METHOD} \
      --seed ${SEED} \
      --logging.log_dir ${LOG_DIR} \
      > "${LOG_DIR}/stdout.log" 2>&1 &
    PIDS+=($!)
  done

  # Wait for all 3 methods to finish before starting next seed
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "  Waiting for ${#PIDS[@]} jobs..."
    for PID in "${PIDS[@]}"; do
      wait $PID
    done
    echo "  Seed ${SEED} complete."
  fi
  echo ""
done

echo "=== All runs complete. Summary: ==="
for METHOD in $METHODS; do
  echo ""
  echo "${METHOD}:"
  for SEED in $SEEDS; do
    CSV="results/halfcheetah/${METHOD}/seed_${SEED}/progress.csv"
    if [[ -f "$CSV" ]]; then
      python3 -c "
import pandas as pd
df = pd.read_csv('${CSV}')
ev = df.dropna(subset=['eval/mean_return'])
tr = df.dropna(subset=['train/mean_reward'])
last_eval = ev['eval/mean_return'].iloc[-1] if len(ev) > 0 else float('nan')
last_train = tr['train/mean_reward'].iloc[-1] if len(tr) > 0 else float('nan')
best_eval = ev['eval/mean_return'].max() if len(ev) > 0 else float('nan')
print(f'  seed={${SEED}}  last_eval={last_eval:.1f}  best_eval={best_eval:.1f}  last_train={last_train:.1f}')
" 2>/dev/null || echo "  seed=${SEED}  (parse error)"
    else
      echo "  seed=${SEED}  (no results)"
    fi
  done
done
