#!/usr/bin/env bash
#SBATCH -J rul_gru_temporal_stage1
#SBATCH --partition=gpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/gru_temporal_stage1.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
SEQUENCE_WINDOW_SIZES="${SEQUENCE_WINDOW_SIZES:-30 45 60}"
DEVICE="${DEVICE:-cuda}"

# Per-subset rolling windows (best ±1 neighbour each side per the spec).
ROLLING_WINDOWS_FD001="${ROLLING_WINDOWS_FD001:-10 15 20}"
ROLLING_WINDOWS_FD002="${ROLLING_WINDOWS_FD002:-10 15 20}"
ROLLING_WINDOWS_FD003="${ROLLING_WINDOWS_FD003:-10 15 20}"
ROLLING_WINDOWS_FD004="${ROLLING_WINDOWS_FD004:-5 10 15}"

ROLLING_FEATURE_SET_FD001="${ROLLING_FEATURE_SET_FD001:-rolling_mean}"
ROLLING_FEATURE_SET_FD002="${ROLLING_FEATURE_SET_FD002:-rolling_mean}"
ROLLING_FEATURE_SET_FD003="${ROLLING_FEATURE_SET_FD003:-rolling_mean}"
ROLLING_FEATURE_SET_FD004="${ROLLING_FEATURE_SET_FD004:-raw_plus_rolling_mean}"

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

read -r -a SEQUENCE_WINDOW_ARGS <<< "$SEQUENCE_WINDOW_SIZES"

echo "project_dir=$PROJECT_DIR"
echo "sequence_window_sizes=${SEQUENCE_WINDOW_ARGS[*]}"
echo "device=$DEVICE"

for fd in fd001 fd002 fd003 fd004; do
    fd_upper="${fd^^}"
    rolling_var="ROLLING_WINDOWS_${fd_upper}"
    feature_var="ROLLING_FEATURE_SET_${fd_upper}"
    read -r -a rolling_args <<< "${!rolling_var}"
    feature_set="${!feature_var}"
    echo ""
    echo "=== ${fd_upper}: feature_set=${feature_set} rolling_windows=${rolling_args[*]} ==="
    turbofan-sweep-gru-temporal \
        --config "configs/subsets/$fd.yaml" \
        --rolling-feature-set "$feature_set" \
        --rolling-windows "${rolling_args[@]}" \
        --sequence-window-sizes "${SEQUENCE_WINDOW_ARGS[@]}" \
        --device "$DEVICE"
done
