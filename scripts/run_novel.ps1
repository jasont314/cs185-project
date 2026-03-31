$HC = "configs/halfcheetah.yaml"
$S = 42
$TB = "--logging.use_tensorboard false"
$NE = 16

foreach ($K in 4, 8, 16) {
    Write-Host "=== K=$K ==="
    $jobs = @()
    foreach ($spec in @(
        @{mode="holistic"; extra=""},
        @{mode="waterfill"; extra=""},
        @{mode="vmpo"; extra=""}
    )) {
        $mode = $spec.mode
        $extra = $spec.extra
        $dir = "results/novel_v2/K${K}_${mode}/seed_$S"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  [START] K=$K $mode"
        $args = "-m src.training.trainer --config $HC --method.mode $mode --policy.K $K --num_envs $NE --seed $S --logging.log_dir $dir $TB $extra"
        $jobs += Start-Process python -ArgumentList $args -NoNewWindow -PassThru
    }
    Write-Host "  Waiting..."
    $jobs | Wait-Process
    Write-Host "  K=$K done."
}

Write-Host ""
Write-Host "=== RESULTS ==="
python scripts/check_results_v2.py
