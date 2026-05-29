#!/usr/bin/env pwsh
# Ridge feature sweep across all four C-MAPSS subsets.
# Outputs: results/feature_sweep_ridge_fd00{1..4}.csv

. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops

$FeatureSets = $env:FEATURE_SETS -split ' ' | Where-Object { $_ } | ForEach-Object { $_ }
if (-not $FeatureSets) { $FeatureSets = @("raw", "rolling_mean", "lag") }

$Windows = $env:WINDOWS -split ' ' | Where-Object { $_ }
if (-not $Windows) { $Windows = @("5", "10", "20") }

$LagSteps = $env:LAG_STEPS -split ' ' | Where-Object { $_ }
if (-not $LagSteps) { $LagSteps = @("2", "4", "8") }

$NJobs = if ($env:N_JOBS) { $env:N_JOBS } else { [Environment]::ProcessorCount }

Write-Host "feature_sets=$($FeatureSets -join ' ')"
Write-Host "windows=$($Windows -join ' ')"
Write-Host "lag_steps=$($LagSteps -join ' ')"
Write-Host "n_jobs=$NJobs"

$failed = @()

foreach ($fd in @("fd001", "fd002", "fd003", "fd004")) {
    Write-Host "`n=== $($fd.ToUpper()) ===" -ForegroundColor Cyan
    turbofan-sweep-features `
        --config "configs/subsets/$fd.yaml" `
        --model ridge `
        --feature-sets @FeatureSets `
        --windows @Windows `
        --lag-steps @LagSteps `
        --n-jobs $NJobs

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$fd failed (exit $LASTEXITCODE)"
        $failed += $fd
    }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All datasets complete." -ForegroundColor Green
} else {
    Write-Warning "Failed: $($failed -join ', ')"
    exit 1
}
