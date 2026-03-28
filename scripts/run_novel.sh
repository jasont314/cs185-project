#!/bin/bash
# Run novel algorithms: KL Waterfilling + Per-Step V-MPO
# Compare against holistic baseline at K=4, 8, 16
set -e
HC=configs/halfcheetah.yaml; S=42; TB="--logging.use_tensorboard false"

run() { local D=$1; shift; mkdir -p "$D"; echo "  [START] $D"; python -m src.training.trainer "$@" --logging.log_dir "$D" $TB > "$D/stdout.log" 2>&1 & }

for K in 4 8 16; do
  echo "=== K=$K ==="
  run results/novel_final/K${K}_holistic/seed_$S    --config $HC --method.mode holistic --policy.K $K --seed $S
  run results/novel_final/K${K}_waterfill/seed_$S   --config $HC --method.mode waterfill --policy.K $K --seed $S
  run results/novel_final/K${K}_vmpo/seed_$S        --config $HC --method.mode vmpo --policy.K $K --seed $S
  echo "  Waiting..."; wait; echo "  K=$K done."
done

echo ""
echo "=== RESULTS ==="
python3 -c "
import pandas as pd, glob
print(f'{\"Config\":<30s} {\"Best\":>8s} {\"Final\":>8s}')
print('-' * 50)
for c in sorted(glob.glob('results/novel_final/*/seed_42/progress.csv')):
    df = pd.read_csv(c)
    ev = df.dropna(subset=['eval/mean_return'])
    n = c.split('novel_final/')[1].split('/seed')[0]
    if len(ev) > 0:
        b = ev['eval/mean_return'].max()
        f = ev['eval/mean_return'].iloc[-1]
        print(f'{n:<30s} {b:>8.1f} {f:>8.1f}')
"
