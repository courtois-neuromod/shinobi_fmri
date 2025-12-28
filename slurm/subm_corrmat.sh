#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_corrmat
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Load configuration from config.yaml
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shinobi_corrmat"

"$PYTHON_BIN" "$SCRIPTS_DIR/correlations/compute_beta_correlations.py"
