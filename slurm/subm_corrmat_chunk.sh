#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=03:00:00
#SBATCH --job-name=shi_corr_chunk
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Create log directory
mkdir -p logfiles/shi_corr_chunk

CHUNK_START=${1:-0}
LOG_DIR=${2:-}
VERBOSE_FLAG=${3:-}
CHUNK_SIZE=100

# Build command with optional arguments
CMD="/home/hyruuk/python_envs/shinobi/bin/python \
    /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/correlations/compute_beta_correlations.py \
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
echo "Starting SLURM job ${SLURM_JOB_ID}"
echo "Chunk start: ${CHUNK_START}"
echo "Log directory: ${LOG_DIR:-'default'}"
echo "Verbosity: ${VERBOSE_FLAG:-'default'}"
echo "Command: $CMD"
echo "========================================"

# Execute the command
eval $CMD
