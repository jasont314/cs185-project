#!/bin/bash
set -e
HC=configs/halfcheetah.yaml; S=42; TB="--logging.use_tensorboard false"

run() { local D=$1; shift; mkdir -p "$D"; echo "  [START] $D"; python -m src.training.trainer "$@" --logging.log_dir "$D" $TB > "$D/stdout.log" 2>&1 & }

for K in 4 8 16; do
  echo "=== K=$K, sigma_init=2.0 ==="
  run results/sigma2/K${K}_holistic/seed_$S     --config $HC --method.mode holistic --policy.K $K --policy.sigma_init 2.0 --seed $S
  run results/sigma2/K${K}_hier_asym/seed_$S    --config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --method.asymmetric_clip true --policy.K $K --policy.sigma_init 2.0 --seed $S
  run results/sigma2/K${K}_cumulative/seed_$S   --config $HC --method.mode cumulative --policy.K $K --policy.sigma_init 2.0 --seed $S
  echo "  Waiting..."; wait; echo "  K=$K done."
done

echo ""
echo "=== RESULTS ==="
python3 -c "
import pandas as pd, glob
for c in sorted(glob.glob('results/sigma2/*/seed_42/progress.csv')):
    df = pd.read_csv(c)
    ev = df.dropna(subset=['eval/mean_return'])
    n = c.split('sigma2/')[1].split('/seed')[0]
    b = ev['eval/mean_return'].max() if len(ev) > 0 else float('nan')
    f = ev['eval/mean_return'].iloc[-1] if len(ev) > 0 else float('nan')
    print(f'{n:<30s} best={b:>8.1f}  final={f:>8.1f}')
"
