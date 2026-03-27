#!/bin/bash
# Hopper experiments — 1 seed, 3 methods in parallel.
# Hopper is harder than HalfCheetah (balance + falls), per-step may help more.

set -e

CONFIG=configs/hopper.yaml
SEED=42

echo "=== Hopper-v5 — Holistic vs Per-Step Comparison ==="
echo "Running 3 methods in parallel..."

PIDS=()

# Holistic baseline
LOG=results/hopper/holistic/seed_${SEED}
mkdir -p $LOG
echo "  [START] holistic"
python -m src.training.trainer --config $CONFIG --method.mode holistic \
  --seed $SEED --logging.log_dir $LOG > ${LOG}/stdout.log 2>&1 &
PIDS+=($!)

# Per-step uniform with 2x clip
LOG=results/hopper/perstep_uniform_clip2x/seed_${SEED}
mkdir -p $LOG
echo "  [START] per_step_uniform (clip_eps_scale=2)"
python -m src.training.trainer --config $CONFIG --method.mode per_step_uniform \
  --seed $SEED --method.clip_eps_scale 2.0 \
  --logging.log_dir $LOG > ${LOG}/stdout.log 2>&1 &
PIDS+=($!)

# Per-step learned global with 2x clip + weight entropy
LOG=results/hopper/perstep_learned_clip2x/seed_${SEED}
mkdir -p $LOG
echo "  [START] per_step_learned_global (clip_eps_scale=2, weight_ent=0.01)"
python -m src.training.trainer --config $CONFIG --method.mode per_step_learned_global \
  --seed $SEED --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 \
  --logging.log_dir $LOG > ${LOG}/stdout.log 2>&1 &
PIDS+=($!)

echo "  Waiting for ${#PIDS[@]} jobs..."
for PID in "${PIDS[@]}"; do
  wait $PID
done

echo ""
echo "=== Hopper Results ==="
python3 -c "
import pandas as pd, glob
for csv in sorted(glob.glob('results/hopper/*/seed_${SEED}/progress.csv')):
    df = pd.read_csv(csv)
    name = csv.split('hopper/')[1].split('/seed')[0]
    ev = df.dropna(subset=['eval/mean_return'])
    if len(ev) == 0: continue
    best = ev['eval/mean_return'].max()
    final = ev['eval/mean_return'].iloc[-1]
    std = ev['eval/std_return'].iloc[-1]
    print(f'  {name:<35s} best={best:>7.1f}  final={final:>7.1f} ± {std:.0f}')
"
