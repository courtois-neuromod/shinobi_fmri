#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Arguments from task launcher:
# $1 = subject (e.g., sub-01)
# $2 = screening (percentile, default: 20)
# $3 = n_jobs (CPUs, default: 40)
# $4 = exclude_low_level (true/false, default: false)

SUBJECT=$1
SCREENING=${2:-20}
N_JOBS=${3:-40}
EXCLUDE_LOW_LEVEL=${4:-false}

# Get repository root - use SLURM_SUBMIT_DIR (directory where sbatch was called)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for local testing
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shi_mvpa_seslvl

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

echo "=========================================="
echo "MVPA Session-Level Decoder"
echo "=========================================="
echo "Subject:          $SUBJECT"
echo "Screening:        $SCREENING%"
echo "CPUs:             $N_JOBS"
echo "Exclude low-level: $EXCLUDE_LOW_LEVEL"
echo "Python:           $PYTHON_BIN"
echo "=========================================="

# Build command arguments
# Add --exclude-low-level flag when EXCLUDE_LOW_LEVEL is true
CMD_ARGS="--subject $SUBJECT --screening $SCREENING --n-jobs $N_JOBS -v"
if [ "$EXCLUDE_LOW_LEVEL" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --exclude-low-level"
fi

# Run MVPA decoder
"$PYTHON_BIN" shinobi_fmri/mvpa/compute_mvpa.py $CMD_ARGS

echo "=========================================="
echo "Decoder completed"
echo "=========================================="
