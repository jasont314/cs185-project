#!/bin/bash
# Ablation experiments — 1 seed each, 3 methods in parallel per group.
# Groups run sequentially to stay within 16GB RAM.
#
# Usage:
#   bash scripts/run_ablations.sh

set -e

CONFIG=configs/halfcheetah.yaml
SEED=42

run_group() {
  local GROUP_NAME="$1"
  shift
  local CMDS=("$@")

  echo ""
  echo "=== ${GROUP_NAME} ==="
  PIDS=()
  for CMD in "${CMDS[@]}"; do
    echo "  [START] ${CMD##*--logging.log_dir }"
    eval "$CMD" &
    PIDS+=($!)
  done
  echo "  Waiting for ${#PIDS[@]} jobs..."
  for PID in "${PIDS[@]}"; do
    wait $PID
  done
  echo "  ${GROUP_NAME} complete."
}

# ---------------------------------------------------------------
# Group 1: Baseline holistic + improved per-step (clip_eps_scale=2)
# ---------------------------------------------------------------
run_group "Improved per-step (clip_eps_scale=2)" \
  "python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --logging.log_dir results/ablations/holistic_baseline/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --method.clip_eps_scale 2.0 --logging.log_dir results/ablations/perstep_uniform_clip2x/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --logging.log_dir results/ablations/perstep_learned_clip2x_went0.01/seed_${SEED} > /dev/null 2>&1"

# ---------------------------------------------------------------
# Group 2: K=2 ablation (all 3 methods)
# ---------------------------------------------------------------
run_group "K=2 ablation" \
  "python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --policy.K 2 --logging.log_dir results/ablations/K2_holistic/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --policy.K 2 --method.clip_eps_scale 2.0 --logging.log_dir results/ablations/K2_perstep_uniform/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --policy.K 2 --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --logging.log_dir results/ablations/K2_perstep_learned/seed_${SEED} > /dev/null 2>&1"

# ---------------------------------------------------------------
# Group 3: K=8 ablation (all 3 methods)
# ---------------------------------------------------------------
run_group "K=8 ablation" \
  "python -m src.training.trainer --config $CONFIG --method.mode holistic --seed $SEED --policy.K 8 --logging.log_dir results/ablations/K8_holistic/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform --seed $SEED --policy.K 8 --method.clip_eps_scale 2.0 --logging.log_dir results/ablations/K8_perstep_uniform/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --policy.K 8 --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --logging.log_dir results/ablations/K8_perstep_learned/seed_${SEED} > /dev/null 2>&1"

# ---------------------------------------------------------------
# Group 4: Weight entropy ablation (K=4, clip_eps_scale=2)
# ---------------------------------------------------------------
run_group "Weight entropy ablation" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.0 --logging.log_dir results/ablations/perstep_learned_went0.0/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.1 --logging.log_dir results/ablations/perstep_learned_went0.1/seed_${SEED} > /dev/null 2>&1" \
  "python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global --seed $SEED --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.001 --logging.log_dir results/ablations/perstep_learned_went0.001/seed_${SEED} > /dev/null 2>&1"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""

python3 -c "
import pandas as pd, os, glob

dirs = sorted(glob.glob('results/ablations/*/seed_${SEED}'))
print(f'{\"Experiment\":<45s} {\"Best Eval\":>10s} {\"Final Eval\":>12s}')
print('-' * 70)
for d in dirs:
    csv = os.path.join(d, 'progress.csv')
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    ev = df.dropna(subset=['eval/mean_return'])
    if len(ev) == 0:
        continue
    name = d.split('ablations/')[1].split('/seed')[0]
    best = ev['eval/mean_return'].max()
    final = ev['eval/mean_return'].iloc[-1]
    std = ev['eval/std_return'].iloc[-1]
    print(f'{name:<45s} {best:>10.1f} {final:>7.1f} ± {std:.0f}')
"
