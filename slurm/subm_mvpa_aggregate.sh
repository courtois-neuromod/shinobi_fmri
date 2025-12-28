#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=1:00:00
#SBATCH --job-name=shi_mvpa_agg
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Arguments from task launcher:
# $1 = subject (e.g., sub-01)
# $2 = n_permutations (total number)
# $3 = screening (percentile)

SUBJECT=$1
N_PERM=$2
SCREENING=${3:-20}

# Load configuration from config.yaml
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shi_mvpa_agg"

echo "=========================================="
echo "MVPA Permutation Aggregation"
echo "=========================================="
echo "Subject:      $SUBJECT"
echo "Permutations: $N_PERM"
echo "Screening:    $SCREENING%"
echo "Python:       $PYTHON_BIN"
echo "=========================================="

# Load exit status tracking utility
source "$SCRIPT_DIR/rename_logs_on_exit.sh"

# Run aggregation and rename logs based on exit status
run_and_rename_logs "$PYTHON_BIN" "$SCRIPTS_DIR/mvpa/aggregate_permutations.py" \
    --subject $SUBJECT \
    --n-permutations $N_PERM \
    --screening $SCREENING

echo "=========================================="
echo "Aggregation completed"
echo "=========================================="
