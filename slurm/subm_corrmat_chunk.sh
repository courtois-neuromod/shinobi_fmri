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

# Load configuration from config.yaml
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm"

CORRELATION_SCRIPT="$SCRIPTS_DIR/correlations/compute_beta_correlations.py"

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

# Load exit status tracking utility
source "$SCRIPT_DIR/rename_logs_on_exit.sh"

# Execute the command and rename logs based on exit status
eval "run_and_rename_logs $CMD"
