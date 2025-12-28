#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_corrmat
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Get repository root (where config.yaml lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Create log directory
mkdir -p logs/slurm/shinobi_corrmat

# Read Python path from config.yaml
PYTHON_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['python']['slurm_bin'])")

# Make it absolute if relative
if [[ ! "$PYTHON_BIN" = /* ]]; then
    PYTHON_BIN="$REPO_ROOT/$PYTHON_BIN"
fi

# Run correlation computation
"$PYTHON_BIN" shinobi_fmri/correlations/compute_beta_correlations.py
