#!/bin/bash
# K ablation on HalfCheetah — test K=2,8,16 with all methods.
# K=4 results already exist from the main ablations.
# Runs 3 methods per K value in parallel, K values sequential.

set -e

CONFIG=configs/halfcheetah.yaml
SEED=42

for K in 2 8 16; do
  echo ""
  echo "=== K=${K} ==="
  PIDS=()

  for METHOD_ARGS in \
    "holistic||" \
    "per_step_uniform|perstep_uniform|--method.clip_eps_scale 2.0" \
    "per_step_learned_global|perstep_learned|--method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01"; do

    IFS='|' read -r MODE LABEL EXTRA <<< "$METHOD_ARGS"
    LABEL=${LABEL:-$MODE}
    LOG="results/K_ablation/K${K}_${LABEL}/seed_${SEED}"
    mkdir -p $LOG

    if [[ -f "${LOG}/checkpoints/final.pt" ]]; then
      echo "  [SKIP] K=${K} ${LABEL}"
      continue
    fi

    echo "  [START] K=${K} ${LABEL}"
    python -m src.training.trainer --config $CONFIG \
      --method.mode $MODE --seed $SEED --policy.K $K $EXTRA \
      --logging.log_dir $LOG > ${LOG}/stdout.log 2>&1 &
    PIDS+=($!)
  done

  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "  Waiting for ${#PIDS[@]} jobs..."
    for PID in "${PIDS[@]}"; do
      wait $PID
    done
  fi
  echo "  K=${K} complete."
done

echo ""
echo "=== K Ablation Results ==="
python3 -c "
import pandas as pd, glob
print(f'{\"Config\":<35s} {\"Best Eval\":>10s} {\"Final Eval\":>12s}')
print('-' * 60)
for csv in sorted(glob.glob('results/K_ablation/*/seed_${SEED}/progress.csv')):
    df = pd.read_csv(csv)
    name = csv.split('K_ablation/')[1].split('/seed')[0]
    ev = df.dropna(subset=['eval/mean_return'])
    if len(ev) == 0: continue
    best = ev['eval/mean_return'].max()
    final = ev['eval/mean_return'].iloc[-1]
    std = ev['eval/std_return'].iloc[-1]
    print(f'{name:<35s} {best:>10.1f} {final:>7.1f} ± {std:.0f}')
"
