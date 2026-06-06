#!/usr/bin/env bash
# Resumable feature-family screen — CPU-only, designed for SLURM clusters.
#
# USAGE
#   sbatch jobs/slurm/run_feature_family_screen.sh
#
# RESUMABILITY
#   The screen CLI writes results to a CSV as each cell completes.
#   If the job is killed by the walltime limit, re-submit the same command;
#   already-completed cells are detected via the CSV skip-set and skipped.
#
# SPLITTING ACROSS JOBS (per-subset parallelism)
#   SUBSETS=FD001 sbatch jobs/slurm/run_feature_family_screen.sh
#   SUBSETS=FD002 sbatch jobs/slurm/run_feature_family_screen.sh
#   ...
#
# SEED NOISE BAND (does NOT re-run seed-42 cells; the skip-set protects them)
#   SEEDS="43 44 45 46" sbatch jobs/slurm/run_feature_family_screen.sh
#
# CLUSTER-SPECIFIC NOTES
#   Partition, time limit, and CPU count are cluster-specific defaults.
#   Override any of them on the sbatch command line, e.g.:
#     sbatch --partition=compute --time=8:00:00 --cpus-per-task=16 \
#            jobs/slurm/run_feature_family_screen.sh

#SBATCH -J rul_feature_screen
# Partition is cluster-specific; override with --partition= on the sbatch command line.
#SBATCH --partition=cpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
# 8 CPUs is reasonable for torch/pandas threading; tune with --cpus-per-task= as needed.
#SBATCH --cpus-per-task=8
# Spec notes a 4-hour node limit; the resumable CSV makes any overrun a non-event.
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/feature_screen.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"

# Full-grid defaults — override via env vars to submit per-subset or per-seed jobs.
# NOTE: CLI --subsets choices are uppercase (FD001..FD004).
SUBSETS="${SUBSETS:-FD001 FD002 FD003 FD004}"
ARCHITECTURES="${ARCHITECTURES:-gru lstm}"
SEEDS="${SEEDS:-42}"

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

read -r -a SUBSET_ARGS       <<< "$SUBSETS"
read -r -a ARCH_ARGS         <<< "$ARCHITECTURES"
read -r -a SEED_ARGS         <<< "$SEEDS"

echo "project_dir=$PROJECT_DIR"
echo "subsets=${SUBSET_ARGS[*]}"
echo "architectures=${ARCH_ARGS[*]}"
echo "seeds=${SEED_ARGS[*]}"

# Single invocation: run_screen handles per-cell iteration and the resume skip-set
# internally, so re-running this script after a walltime kill safely skips done cells.
# results-dir and configs-dir use their CLI defaults (results/ and configs/subsets/).
turbofan-feature-screen \
    --subsets "${SUBSET_ARGS[@]}" \
    --architectures "${ARCH_ARGS[@]}" \
    --seeds "${SEED_ARGS[@]}"
