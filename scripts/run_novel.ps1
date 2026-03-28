$HC = "configs/halfcheetah.yaml"
$S = 42
$TB = "--logging.use_tensorboard false"
$NE = 16

foreach ($K in 4, 8, 16) {
    Write-Host "=== K=$K ==="
    $jobs = @()
    foreach ($mode in "holistic", "waterfill", "vmpo") {
        $dir = "results/novel_final/K${K}_${mode}/seed_$S"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  [START] $mode"
        $jobs += Start-Process python -ArgumentList "-m src.training.trainer --config $HC --method.mode $mode --policy.K $K --num_envs $NE --seed $S --logging.log_dir $dir $TB" -NoNewWindow -PassThru
    }
    Write-Host "  Waiting..."
    $jobs | Wait-Process
    Write-Host "  K=$K done."
}

Write-Host ""
Write-Host "=== RESULTS ==="
python scripts/check_results.py
