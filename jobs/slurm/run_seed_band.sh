#!/usr/bin/env bash
# Resumable seed-noise band for the feature-family screen — GPU by default,
# runs on CPU nodes unchanged.
#
# WHAT THIS DOES
#   Re-runs only the eight winning cells of the primary screen (one per
#   architecture x subset, ranked by validation RMSE) across several seeds, and
#   appends every result to results/feature_family_seed_band.csv. The within-cell
#   standard deviation of val_rmse across seeds is the empirical seed-noise band.
#   The primary screen CSVs (feature_family_screen_*.csv) are NOT touched.
#
# USAGE
#   sbatch jobs/slurm/run_seed_band.sh
#   SEEDS="43 44 45 46" sbatch jobs/slurm/run_seed_band.sh   # skip seed 42
#
# DEVICE (GPU and CPU nodes, same script)
#   DEVICE defaults to "auto": CUDA when a GPU is visible, else CPU. Pin with
#   DEVICE=cpu or DEVICE=cuda (cuda errors when no GPU is present).
#
# RUNNING ON A CPU NODE
#   sbatch --partition=cpu_short --gres=NONE jobs/slurm/run_seed_band.sh
#
# RESUMABILITY
#   Each completed cell is appended and flushed immediately. If the walltime
#   limit kills the job, re-submit the same command; completed (arch, subset,
#   config, window, lag, sequence, seed) cells are detected and skipped.

#SBATCH -J rul_seed_band
# Partition is cluster-specific; override with --partition= on the sbatch command line.
#SBATCH --partition=cluster_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/seed_band.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
DEVICE="${DEVICE:-auto}"
# Five seeds by default (42 reproduces the primary-screen value). Override to
# skip 42 or add more: SEEDS="43 44 45 46".
SEEDS="${SEEDS:-42 43 44 45 46}"

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

module load cuda
conda activate "$CONDA_ENV"

read -r -a SEED_ARGS <<< "$SEEDS"

echo "project_dir=$PROJECT_DIR"
echo "seeds=${SEED_ARGS[*]}"
echo "device=$DEVICE"

# Standalone driver (no console-script registration needed); imports the
# installed turbofan package and reuses run_cell() so every fixed
# hyperparameter matches the primary screen.
python "$PROJECT_DIR/jobs/slurm/seed_band.py" \
    --seeds "${SEED_ARGS[@]}" \
    --device "$DEVICE"
