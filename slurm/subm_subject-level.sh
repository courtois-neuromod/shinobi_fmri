#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=01:00:00
#SBATCH --job-name=shi_subjlevel
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Load configuration from config.yaml
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shi_subjlevel"

"$PYTHON_BIN" "$SCRIPTS_DIR/glm/compute_subject_level.py" --subject $1 --condition $2
