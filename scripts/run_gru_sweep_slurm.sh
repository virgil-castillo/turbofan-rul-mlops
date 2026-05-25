#!/usr/bin/env bash
#SBATCH -J rul_gru_sweep
#SBATCH --partition=cluster_long
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --gres=gpu:1
#SBATCH --time=4-04:00:00
#SBATCH --output=outputs/logs/gru_sweep.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_SIZES="${WINDOW_SIZES:-15 20 30 45}"
HIDDEN_SIZES="${HIDDEN_SIZES:-32 64 128}"
LEARNING_RATES="${LEARNING_RATES:-1e-3 5e-4 1e-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT="${OUTPUT:-artifacts/gru_sweep_${SLURM_JOB_ID:-local}.csv}"

cd "$PROJECT_DIR"
mkdir -p artifacts outputs/logs

if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
    # Linux/macOS conda layout.
    source "$CONDA_HOME/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_HOME/Scripts/activate" ]]; then
    # Windows/Git-Bash conda layout, useful for local syntax checks.
    source "$CONDA_HOME/Scripts/activate"
else
    echo "Could not find conda activation script under CONDA_HOME=$CONDA_HOME" >&2
    exit 1
fi

conda activate "$CONDA_ENV"

read -r -a WINDOW_ARGS <<< "$WINDOW_SIZES"
read -r -a HIDDEN_SIZE_ARGS <<< "$HIDDEN_SIZES"
read -r -a LEARNING_RATE_ARGS <<< "$LEARNING_RATES"

echo "project_dir=$PROJECT_DIR"
echo "config=$CONFIG"
echo "window_sizes=${WINDOW_ARGS[*]}"
echo "hidden_sizes=${HIDDEN_SIZE_ARGS[*]}"
echo "learning_rates=${LEARNING_RATE_ARGS[*]}"
echo "device=$DEVICE"
echo "output=$OUTPUT"

python scripts/sweep_sequence_gru.py \
    --config "$CONFIG" \
    --window-sizes "${WINDOW_ARGS[@]}" \
    --hidden-sizes "${HIDDEN_SIZE_ARGS[@]}" \
    --learning-rates "${LEARNING_RATE_ARGS[@]}" \
    --device "$DEVICE" \
    --output "$OUTPUT"
