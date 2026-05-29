#!/usr/bin/env bash
#SBATCH -J rul_feature_sweep_gru
#SBATCH --partition=gpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/feature_sweep_gru.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
FEATURE_SETS="${FEATURE_SETS:-raw rolling_mean lag}"
WINDOWS="${WINDOWS:-5 10 20}"
LAG_STEPS="${LAG_STEPS:-2 4 8}"
DEVICE="${DEVICE:-cuda}"

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

read -r -a FEATURE_SET_ARGS <<< "$FEATURE_SETS"
read -r -a WINDOW_ARGS <<< "$WINDOWS"
read -r -a LAG_STEP_ARGS <<< "$LAG_STEPS"

echo "project_dir=$PROJECT_DIR"
echo "feature_sets=${FEATURE_SET_ARGS[*]}"
echo "windows=${WINDOW_ARGS[*]}"
echo "lag_steps=${LAG_STEP_ARGS[*]}"
echo "device=$DEVICE"

for fd in fd001 fd002 fd003 fd004; do
    echo ""
    echo "=== ${fd^^} ==="
    turbofan-sweep-features \
        --config "configs/subsets/$fd.yaml" \
        --model gru \
        --feature-sets "${FEATURE_SET_ARGS[@]}" \
        --windows "${WINDOW_ARGS[@]}" \
        --lag-steps "${LAG_STEP_ARGS[@]}" \
        --device "$DEVICE"
done
