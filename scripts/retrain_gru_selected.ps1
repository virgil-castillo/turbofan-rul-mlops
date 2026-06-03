#!/usr/bin/env pwsh
# Retrain the final (production) GRU model for each C-MAPSS subset using the
# Stage 2 capacity-sweep selected configs encoded in configs/subsets/<fd>.yaml.
#
# Runs on CPU: configs/default.yaml sets `sequence.device: cpu`, and the train
# CLI is config-driven (it takes only --config). Each run writes a timestamped
# artifact dir under artifacts/models/ and logs an MLflow run to mlflow.db
# (turbofan-training experiment).
#
# Override the subset list with the SUBSETS env var, e.g.:
#   $env:SUBSETS = "fd003"; ./scripts/retrain_gru_selected.ps1

. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops

$Subsets = $env:SUBSETS -split ' ' | Where-Object { $_ }
if (-not $Subsets) { $Subsets = @("fd001", "fd002", "fd003", "fd004") }

Write-Host "subsets=$($Subsets -join ' ')"
Write-Host "device=cpu (from configs/default.yaml)"

$failed = @()

foreach ($fd in $Subsets) {
    Write-Host "`n=== retraining $($fd.ToUpper()) ===" -ForegroundColor Cyan
    turbofan-train-sequence-gru --config "configs/subsets/$fd.yaml"

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$fd failed (exit $LASTEXITCODE)"
        $failed += $fd
    }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All GRU models retrained." -ForegroundColor Green
} else {
    Write-Warning "Failed: $($failed -join ', ')"
    exit 1
}
