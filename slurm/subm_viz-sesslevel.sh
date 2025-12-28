#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_viz_seslvl
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Get repository root (where config.yaml lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shi_viz_seslvl

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

# Run visualization
"$PYTHON_BIN" shinobi_fmri/visualization/viz_session-level.py
