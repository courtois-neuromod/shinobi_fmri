#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=03:00:00
#SBATCH --job-name=shi_corr_chunk
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Create log directory
mkdir -p logs/slurm

CHUNK_START=${1:-0}
LOG_DIR=${2:-}
VERBOSE_FLAG=${3:-}
CHUNK_SIZE=100

# Find repo root using SLURM_SUBMIT_DIR (where sbatch was called from)
# Try SLURM_SUBMIT_DIR first (in case called from repo root)
if [ -d "${SLURM_SUBMIT_DIR}/shinobi_fmri" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    # Otherwise assume called from slurm/ subdirectory
    REPO_ROOT="${SLURM_SUBMIT_DIR}/.."
fi

CORRELATION_SCRIPT="${REPO_ROOT}/shinobi_fmri/correlations/compute_beta_correlations.py"

# Verify script exists
if [ ! -f "$CORRELATION_SCRIPT" ]; then
    echo "ERROR: Script not found at: $CORRELATION_SCRIPT"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
    echo "REPO_ROOT: ${REPO_ROOT}"
    echo "Checked for shinobi_fmri/ directory in: ${SLURM_SUBMIT_DIR}"
    exit 1
fi

# Determine Python executable from config.yaml (slurm_bin setting)
# Falls back to 'python' from PATH if not configured
PYTHON_BIN=${SHINOBI_SLURM_PYTHON_BIN:-python}

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
eval $CMD
