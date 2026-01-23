#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_mvpa_perm
#SBATCH --output=logs/slurm/%x/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x/%x_%A_%a.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Arguments from batch launcher:
# $1 = subject (e.g., sub-01)
# $2 = n_permutations (total number)
# $3 = perm_start (starting index for this job)
# $4 = perm_end (ending index for this job)
# $5 = screening (percentile)
# $6 = n_jobs (CPUs)
# $7 = exclude_low_level (true/false, default: false)

SUBJECT=$1
N_PERM=$2
PERM_START=$3
PERM_END=$4
SCREENING=${5:-20}
N_JOBS=${6:-40}
EXCLUDE_LOW_LEVEL=${7:-false}

# Get repository root - use SLURM_SUBMIT_DIR (directory where sbatch was called)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for local testing
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shi_mvpa_perm

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

echo "=========================================="
echo "MVPA Permutation Testing"
echo "=========================================="
echo "Subject:          $SUBJECT"
echo "Permutations:     $PERM_START to $((PERM_END-1)) (out of $N_PERM total)"
echo "Screening:        $SCREENING%"
echo "CPUs:             $N_JOBS"
echo "Exclude low-level: $EXCLUDE_LOW_LEVEL"
echo "Python:           $PYTHON_BIN"
echo "=========================================="

# Add --exclude-low-level flag when EXCLUDE_LOW_LEVEL is true
CMD_ARGS="--subject $SUBJECT --n-permutations $N_PERM --perm-start $PERM_START --perm-end $PERM_END --screening $SCREENING --n-jobs $N_JOBS -v"
if [ "$EXCLUDE_LOW_LEVEL" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --exclude-low-level"
fi

# Run permutation test
"$PYTHON_BIN" shinobi_fmri/mvpa/compute_mvpa.py $CMD_ARGS

echo "=========================================="
echo "Permutation job completed"
echo "=========================================="
