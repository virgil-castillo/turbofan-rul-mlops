#!/usr/bin/env bash
#SBATCH -J rul_gru_selected_retrain
#SBATCH --partition=gpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/gru_selected_retrain.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
DEVICE="${DEVICE:-cuda}"
SUBSETS="${SUBSETS:-fd001 fd002 fd003 fd004}"

cd "$PROJECT_DIR"
mkdir -p results outputs/logs artifacts/models

if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_HOME/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_HOME/Scripts/activate" ]]; then
    source "$CONDA_HOME/Scripts/activate"
else
    echo "Could not find conda activation script under CONDA_HOME=$CONDA_HOME" >&2
    exit 1
fi

conda activate "$CONDA_ENV"

read -r -a SUBSET_ARGS <<< "$SUBSETS"

echo "project_dir=$PROJECT_DIR"
echo "subsets=${SUBSET_ARGS[*]}"
echo "device=$DEVICE"

# Assumes configs/subsets/<fd>.yaml has been updated with the
# Stage 1/2 selected gru block (feature_families, windows, sequence.window_size,
# sequence.hidden_size, sequence.learning_rate). This script just retrains
# each subset's GRU using the existing training CLI.
for fd in "${SUBSET_ARGS[@]}"; do
    echo ""
    echo "=== retraining ${fd^^} ==="
    turbofan-train-sequence-gru \
        --config "configs/subsets/$fd.yaml" \
        --device "$DEVICE"
done
