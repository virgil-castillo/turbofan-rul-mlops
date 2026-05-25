#!/usr/bin/env bash
#SBATCH -J rul_feat_cmp
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/baseline_feature_comparison.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
CONFIG="${CONFIG:-configs/default.yaml}"
FEATURE_SETS="${FEATURE_SETS:-raw raw_plus_rolling rolling}"
WINDOWS="${WINDOWS:-5 10 20}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT="${OUTPUT:-artifacts/baseline_feature_comparison_${SLURM_JOB_ID:-local}.csv}"

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

read -r -a FEATURE_SET_ARGS <<< "$FEATURE_SETS"
read -r -a WINDOW_ARGS <<< "$WINDOWS"

echo "project_dir=$PROJECT_DIR"
echo "config=$CONFIG"
echo "feature_sets=${FEATURE_SET_ARGS[*]}"
echo "windows=${WINDOW_ARGS[*]}"
echo "n_jobs=$N_JOBS"
echo "output=$OUTPUT"

python scripts/compare_baseline_features.py \
    --config "$CONFIG" \
    --feature-sets "${FEATURE_SET_ARGS[@]}" \
    --windows "${WINDOW_ARGS[@]}" \
    --n-jobs "$N_JOBS" \
    --output "$OUTPUT"
