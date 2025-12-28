#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_viz_sublvl
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

# Get repository root - use SLURM_SUBMIT_DIR (directory where sbatch was called)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for local testing
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shi_viz_sublvl

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

# Run visualization
"$PYTHON_BIN" shinobi_fmri/visualization/viz_subject-level.py
