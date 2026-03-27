#!/bin/bash
# 8-hour experiment plan: 16 groups of 3 parallel runs
# Total: 48 experiments across HalfCheetah and Hopper, K=2,4,8,16
set -e

HC=configs/halfcheetah.yaml
HP=configs/hopper.yaml
S=42
TB="--logging.use_tensorboard false"

run_group() {
  local NAME="$1"; shift
  echo ""
  echo "=== $NAME ==="
  PIDS=()
  for CMD in "$@"; do
    local DIR=$(echo "$CMD" | grep -o 'log_dir [^ ]*' | cut -d' ' -f2)
    if [[ -f "${DIR}/checkpoints/final.pt" ]]; then
      echo "  [SKIP] $DIR"
      continue
    fi
    mkdir -p "$DIR"
    echo "  [START] $DIR"
    eval "python -m src.training.trainer $CMD" > "${DIR}/stdout.log" 2>&1 &
    PIDS+=($!)
  done
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "  Waiting for ${#PIDS[@]} jobs..."
    for P in "${PIDS[@]}"; do wait $P; done
  fi
  echo "  $NAME done."
}

# ====================================================================
# PHASE 1: K=16 -- Closest to beating holistic (981)
# ====================================================================

run_group "G1: K=16 core methods" \
  "--config $HC --method.mode hierarchical --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K16_hierarchical/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K16_kl_inverse/seed_$S $TB" \
  "--config $HC --method.mode cumulative --policy.K 16 --seed $S --logging.log_dir results/8hr/K16_cumulative/seed_$S $TB"

run_group "G2: K=16 combos + secondary" \
  "--config $HC --method.mode hierarchical_cumulative --policy.K 16 --seed $S --logging.log_dir results/8hr/K16_hier_cum/seed_$S $TB" \
  "--config $HC --method.mode hierarchical --policy.K 16 --method.clip_eps_scale 2.0 --method.asymmetric_clip true --seed $S --logging.log_dir results/8hr/K16_hier_asym/seed_$S $TB" \
  "--config $HC --method.mode step_conditioned --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K16_step_cond/seed_$S $TB"

run_group "G3: K=16 clip scale + beta sweep" \
  "--config $HC --method.mode per_step_learned_global --policy.K 16 --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --seed $S --logging.log_dir results/8hr/K16_learned_clip2x/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --policy.K 16 --method.clip_eps_scale 2.0 --method.kl_beta 1.0 --seed $S --logging.log_dir results/8hr/K16_kl_beta1/seed_$S $TB" \
  "--config $HC --method.mode cumulative --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K16_cum_clip2x/seed_$S $TB"

run_group "G4: K=16 best tuned" \
  "--config $HC --method.mode hierarchical --policy.K 16 --method.clip_eps_scale 2.0 --method.delta_reg 1.0 --seed $S --logging.log_dir results/8hr/K16_hier_medreg/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --policy.K 16 --method.clip_eps_scale 2.0 --method.kl_beta 10.0 --seed $S --logging.log_dir results/8hr/K16_kl_beta10/seed_$S $TB" \
  "--config $HC --method.mode intra_chain --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K16_intra/seed_$S $TB"

# ====================================================================
# PHASE 2: K=4 -- Most data for comparison (holistic=848)
# ====================================================================

run_group "G5: K=4 core new methods" \
  "--config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K4_hierarchical/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K4_kl_inverse/seed_$S $TB" \
  "--config $HC --method.mode cumulative --seed $S --logging.log_dir results/8hr/K4_cumulative/seed_$S $TB"

run_group "G6: K=4 combos + secondary" \
  "--config $HC --method.mode hierarchical_cumulative --seed $S --logging.log_dir results/8hr/K4_hier_cum/seed_$S $TB" \
  "--config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --method.asymmetric_clip true --seed $S --logging.log_dir results/8hr/K4_hier_asym/seed_$S $TB" \
  "--config $HC --method.mode step_conditioned --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K4_step_cond/seed_$S $TB"

run_group "G7: K=4 rescue per-step" \
  "--config $HC --method.mode per_step_learned_global --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --seed $S --logging.log_dir results/8hr/K4_learned_clip2x/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --method.clip_eps_scale 2.0 --method.kl_beta 1.0 --seed $S --logging.log_dir results/8hr/K4_kl_beta1/seed_$S $TB" \
  "--config $HC --method.mode intra_chain --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K4_intra_clip2x/seed_$S $TB"

run_group "G8: K=4 hierarchical reg sweep" \
  "--config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --method.delta_reg 1.0 --seed $S --logging.log_dir results/8hr/K4_hier_medreg/seed_$S $TB" \
  "--config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --method.delta_reg 0.01 --seed $S --logging.log_dir results/8hr/K4_hier_tinyreg/seed_$S $TB" \
  "--config $HC --method.mode cumulative --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K4_cum_clip2x/seed_$S $TB"

# ====================================================================
# PHASE 3: Hopper -- Generalization
# ====================================================================

run_group "G9: Hopper K=4 core" \
  "--config $HP --method.mode hierarchical --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K4_hierarchical/seed_$S $TB" \
  "--config $HP --method.mode per_step_kl_inverse --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K4_kl_inverse/seed_$S $TB" \
  "--config $HP --method.mode cumulative --seed $S --logging.log_dir results/8hr/hop_K4_cumulative/seed_$S $TB"

run_group "G10: Hopper K=4 secondary" \
  "--config $HP --method.mode step_conditioned --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K4_step_cond/seed_$S $TB" \
  "--config $HP --method.mode per_step_uniform --method.clip_eps_scale 2.0 --method.asymmetric_clip true --seed $S --logging.log_dir results/8hr/hop_K4_asym/seed_$S $TB" \
  "--config $HP --method.mode per_step_learned_global --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --seed $S --logging.log_dir results/8hr/hop_K4_learned/seed_$S $TB"

run_group "G11: Hopper K=16" \
  "--config $HP --method.mode holistic --policy.K 16 --seed $S --logging.log_dir results/8hr/hop_K16_holistic/seed_$S $TB" \
  "--config $HP --method.mode hierarchical --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K16_hierarchical/seed_$S $TB" \
  "--config $HP --method.mode per_step_kl_inverse --policy.K 16 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K16_kl_inverse/seed_$S $TB"

run_group "G12: Hopper K=16 secondary" \
  "--config $HP --method.mode cumulative --policy.K 16 --seed $S --logging.log_dir results/8hr/hop_K16_cumulative/seed_$S $TB" \
  "--config $HP --method.mode per_step_learned_global --policy.K 16 --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --seed $S --logging.log_dir results/8hr/hop_K16_learned/seed_$S $TB" \
  "--config $HP --method.mode intra_chain --policy.K 4 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/hop_K4_intra/seed_$S $TB"

# ====================================================================
# PHASE 4: K=2, K=8 sweeps + fills
# ====================================================================

run_group "G13: K=2 and K=8 hierarchical" \
  "--config $HC --method.mode hierarchical --policy.K 2 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K2_hierarchical/seed_$S $TB" \
  "--config $HC --method.mode hierarchical --policy.K 8 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K8_hierarchical/seed_$S $TB" \
  "--config $HC --method.mode per_step_kl_inverse --policy.K 8 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K8_kl_inverse/seed_$S $TB"

run_group "G14: K=8 why collapse?" \
  "--config $HC --method.mode cumulative --policy.K 8 --seed $S --logging.log_dir results/8hr/K8_cumulative/seed_$S $TB" \
  "--config $HC --method.mode per_step_learned_global --policy.K 8 --method.clip_eps_scale 2.0 --method.weight_ent_coef 0.01 --seed $S --logging.log_dir results/8hr/K8_learned_clip2x/seed_$S $TB" \
  "--config $HC --method.mode step_conditioned --policy.K 8 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K8_step_cond/seed_$S $TB"

run_group "G15: K=2 fills" \
  "--config $HC --method.mode per_step_kl_inverse --policy.K 2 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K2_kl_inverse/seed_$S $TB" \
  "--config $HC --method.mode cumulative --policy.K 2 --seed $S --logging.log_dir results/8hr/K2_cumulative/seed_$S $TB" \
  "--config $HC --method.mode step_conditioned --policy.K 2 --method.clip_eps_scale 2.0 --seed $S --logging.log_dir results/8hr/K2_step_cond/seed_$S $TB"

run_group "G16: Seed reruns of best" \
  "--config $HC --method.mode hierarchical --policy.K 16 --method.clip_eps_scale 2.0 --seed 123 --logging.log_dir results/8hr/K16_hierarchical/seed_123 $TB" \
  "--config $HC --method.mode hierarchical --method.clip_eps_scale 2.0 --seed 123 --logging.log_dir results/8hr/K4_hierarchical/seed_123 $TB" \
  "--config $HP --method.mode hierarchical --method.clip_eps_scale 2.0 --seed 123 --logging.log_dir results/8hr/hop_K4_hierarchical/seed_123 $TB"

# ====================================================================
# SUMMARY
# ====================================================================
echo ""
echo "=== FINAL RESULTS ==="
python3 -c "
import pandas as pd, glob, os
results = []
for csv in sorted(glob.glob('results/8hr/*/seed_*/progress.csv')):
    df = pd.read_csv(csv)
    ev = df.dropna(subset=['eval/mean_return'])
    if len(ev) == 0: continue
    name = csv.split('8hr/')[1].replace('/seed_42/progress.csv','').replace('/seed_123/progress.csv',' (s123)')
    best = ev['eval/mean_return'].max()
    final = ev['eval/mean_return'].iloc[-1]
    results.append((name, best, final))

results.sort(key=lambda x: -x[1])
print(f'{\"Method\":<45s} {\"Best\":>8s} {\"Final\":>8s}')
print('-' * 65)
for name, best, final in results:
    print(f'{name:<45s} {best:>8.1f} {final:>8.1f}')
"
