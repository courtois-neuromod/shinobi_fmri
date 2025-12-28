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

# Get repository root (where config.yaml lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shi_mvpa_agg

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

echo "=========================================="
echo "MVPA Permutation Aggregation"
echo "=========================================="
echo "Subject:      $SUBJECT"
echo "Permutations: $N_PERM"
echo "Screening:    $SCREENING%"
echo "Python:       $PYTHON_BIN"
echo "=========================================="

# Run aggregation
"$PYTHON_BIN" shinobi_fmri/mvpa/aggregate_permutations.py \
    --subject $SUBJECT \
    --n-permutations $N_PERM \
    --screening $SCREENING

echo "=========================================="
echo "Aggregation completed"
echo "=========================================="
