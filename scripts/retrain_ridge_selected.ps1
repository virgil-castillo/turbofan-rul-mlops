#!/usr/bin/env pwsh
# Retrain the final (production) Ridge model for each C-MAPSS subset using the
# selected feature config encoded in the features.ridge block of
# configs/subsets/<fd>.yaml.
#
# Runs on CPU (Ridge is CPU-only). The train CLI is config-driven (it takes only
# --config) and performs official-test evaluation when the test files are present.
# Each run writes a timestamped artifact dir under artifacts/models/baseline/ and
# appends a row to results/training_log.jsonl. Ridge is fast — seconds per subset.
#
# Override the subset list with the SUBSETS env var, e.g.:
#   $env:SUBSETS = "fd003"; ./scripts/retrain_ridge_selected.ps1

. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops

$Subsets = $env:SUBSETS -split ' ' | Where-Object { $_ }
if (-not $Subsets) { $Subsets = @("fd001", "fd002", "fd003", "fd004") }

Write-Host "subsets=$($Subsets -join ' ')"
Write-Host "model=ridge (CPU)"

$failed = @()

foreach ($fd in $Subsets) {
    Write-Host "`n=== retraining $($fd.ToUpper()) ===" -ForegroundColor Cyan
    turbofan-train-baseline --config "configs/subsets/$fd.yaml"

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$fd failed (exit $LASTEXITCODE)"
        $failed += $fd
    }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All Ridge models retrained." -ForegroundColor Green
} else {
    Write-Warning "Failed: $($failed -join ', ')"
    exit 1
}
