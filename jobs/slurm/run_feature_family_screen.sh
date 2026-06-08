#!/usr/bin/env bash
# Resumable feature-family screen — GPU by default, runs on CPU nodes unchanged.
#
# USAGE
#   sbatch jobs/slurm/run_feature_family_screen.sh
#
# DEVICE (GPU and CPU nodes, same script)
#   DEVICE defaults to "auto": the screen CLI selects CUDA when a GPU is visible
#   and silently falls back to CPU otherwise. So this one script runs on both
#   node types — no edits needed. Pin it with DEVICE=cpu or DEVICE=cuda if you
#   want to force a device (DEVICE=cuda errors when no GPU is present).
#
# RUNNING ON A CPU NODE
#   The #SBATCH defaults below request a GPU partition. To submit to a CPU node,
#   override the partition and drop the GPU request on the command line:
#     sbatch --partition=cpu_short --gres=NONE \
#            jobs/slurm/run_feature_family_screen.sh
#   DEVICE=auto then resolves to CPU on that node automatically.
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
#   Partition, time limit, and CPU/GPU counts are cluster-specific defaults.
#   Override any of them on the sbatch command line, e.g.:
#     sbatch --partition=gpu_long --time=8:00:00 --cpus-per-task=16 \
#            jobs/slurm/run_feature_family_screen.sh

#SBATCH -J rul_feature_screen
# Partition is cluster-specific; override with --partition= on the sbatch command line.
#SBATCH --partition=gpu_short
#SBATCH -N 1
#SBATCH --ntasks=1
# Request one GPU; ignored on CPU nodes if you override with --gres=NONE.
#SBATCH --gres=gpu:1
# Change to 48 for cluster_short, which has 48-CPU nodes; 52 for cluster_gpu_long, which has 52-CPU nodes.
#SBATCH --cpus-per-task=52
# Spec notes a 4-hour node limit; the resumable CSV makes any overrun a non-event.
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/feature_screen.%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mlops}"
# "auto" = use CUDA if a GPU is visible, else CPU. The same script then runs on
# both GPU and CPU nodes. Pin to cpu/cuda to force a device.
DEVICE="${DEVICE:-auto}"

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

module load cuda
conda activate "$CONDA_ENV"

read -r -a SUBSET_ARGS       <<< "$SUBSETS"
read -r -a ARCH_ARGS         <<< "$ARCHITECTURES"
read -r -a SEED_ARGS         <<< "$SEEDS"

echo "project_dir=$PROJECT_DIR"
echo "subsets=${SUBSET_ARGS[*]}"
echo "architectures=${ARCH_ARGS[*]}"
echo "seeds=${SEED_ARGS[*]}"
echo "device=$DEVICE"

# Single invocation: run_screen handles per-cell iteration and the resume skip-set
# internally, so re-running this script after a walltime kill safely skips done cells.
# results-dir and configs-dir use their CLI defaults (results/ and configs/subsets/).
# --device passes DEVICE through; "auto" resolves per node (CUDA if present, else CPU).
turbofan-feature-screen \
    --subsets "${SUBSET_ARGS[@]}" \
    --architectures "${ARCH_ARGS[@]}" \
    --seeds "${SEED_ARGS[@]}" \
    --device "$DEVICE"
