#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_viz_sublvl
#SBATCH --output=logs/slurm/%x/%x_%j.out
#SBATCH --error=logs/slurm/%x/%x_%j.err
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

# Load configuration and utilities
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/load_config.sh"
source "$SCRIPT_DIR/rename_logs_on_exit.sh"

# Create log directory
mkdir -p "$LOGS_DIR/slurm/shi_viz_sublvl"

# Run visualization and rename logs based on exit status
run_and_rename_logs "$PYTHON_BIN" "$SCRIPTS_DIR/visualization/viz_subject-level.py"
