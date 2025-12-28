#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=03:00:00
#SBATCH --job-name=shi_corr_chunk
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

CHUNK_START=${1:-0}
LOG_DIR=${2:-}
VERBOSE_FLAG=${3:-}
CHUNK_SIZE=100

# Get repository root - use SLURM_SUBMIT_DIR (directory where sbatch was called)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for local testing
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

CORRELATION_SCRIPT="shinobi_fmri/correlations/compute_beta_correlations.py"

# Verify script exists
if [ ! -f "$CORRELATION_SCRIPT" ]; then
    echo "ERROR: Script not found at: $CORRELATION_SCRIPT"
    echo "REPO_ROOT: ${REPO_ROOT}"
    exit 1
fi

# Build command with optional arguments
CMD="${PYTHON_BIN} \
    ${CORRELATION_SCRIPT} \
    --chunk-size ${CHUNK_SIZE} \
    --chunk-start ${CHUNK_START} \
    --n-jobs 40"

# Add log-dir if provided
if [ -n "$LOG_DIR" ]; then
    CMD="$CMD --log-dir $LOG_DIR"
fi

# Add verbose flag if provided
if [ -n "$VERBOSE_FLAG" ]; then
    CMD="$CMD $VERBOSE_FLAG"
fi

# Print the command for debugging
echo "========================================"
echo "Starting SLURM job ${SLURM_JOB_ID}"
echo "Python: ${PYTHON_BIN}"
echo "Script: ${CORRELATION_SCRIPT}"
echo "Chunk start: ${CHUNK_START}"
echo "Log directory: ${LOG_DIR:-'default'}"
echo "Verbosity: ${VERBOSE_FLAG:-'default'}"
echo "Command: $CMD"
echo "========================================"

# Execute the command
eval "$CMD"
