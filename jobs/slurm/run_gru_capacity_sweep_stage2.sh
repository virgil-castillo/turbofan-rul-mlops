#!/usr/bin/env bash
#SBATCH -J rul_gru_capacity_stage2
#SBATCH --partition=gpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/gru_capacity_stage2.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
TOP_K="${TOP_K:-2}"
HIDDEN_SIZES="${HIDDEN_SIZES:-32 64 128}"
LEARNING_RATES="${LEARNING_RATES:-0.001 0.0003}"
DEVICE="${DEVICE:-cuda}"
STAGE1_DIR="${STAGE1_DIR:-results}"

cd "$PROJECT_DIR"
mkdir -p results outputs/logs

if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_HOME/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_HOME/Scripts/activate" ]]; then
    source "$CONDA_HOME/Scripts/activate"
else
    echo "Could not find conda activation script under CONDA_HOME=$CONDA_HOME" >&2
    exit 1
fi

conda activate "$CONDA_ENV"

read -r -a HIDDEN_ARGS <<< "$HIDDEN_SIZES"
read -r -a LR_ARGS <<< "$LEARNING_RATES"

echo "project_dir=$PROJECT_DIR"
echo "top_k=$TOP_K"
echo "hidden_sizes=${HIDDEN_ARGS[*]}"
echo "learning_rates=${LR_ARGS[*]}"
echo "stage1_dir=$STAGE1_DIR"
echo "device=$DEVICE"

for fd in fd001 fd002 fd003 fd004; do
    stage1_csv="$STAGE1_DIR/stage1_temporal_sweep_${fd}.csv"
    if [[ ! -f "$stage1_csv" ]]; then
        echo "missing Stage 1 CSV for ${fd^^}: $stage1_csv" >&2
        exit 1
    fi
    echo ""
    echo "=== ${fd^^}: stage1_csv=$stage1_csv ==="
    turbofan-sweep-gru-capacity \
        --config "configs/subsets/$fd.yaml" \
        --stage1-csv "$stage1_csv" \
        --top-k "$TOP_K" \
        --hidden-sizes "${HIDDEN_ARGS[@]}" \
        --learning-rates "${LR_ARGS[@]}" \
        --device "$DEVICE"
done
