#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_sesslevel
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Load configuration and utilities
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"
source "$SCRIPT_DIR/rename_logs_on_exit.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shi_sesslevel"

# Run command and rename logs based on exit status
run_and_rename_logs "$PYTHON_BIN" "$SCRIPTS_DIR/glm/compute_session_level.py" --subject $1 --session $2
