#!/bin/bash
# Load configuration from config.yaml for SLURM scripts
# This script should be sourced, not executed: source load_config.sh

# Determine repository root
# If submitted from slurm/ directory, go up one level
if [ -d "$SLURM_SUBMIT_DIR/shinobi_fmri" ]; then
    export REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "$SLURM_SUBMIT_DIR/../shinobi_fmri" ]; then
    export REPO_ROOT="$SLURM_SUBMIT_DIR/.."
else
    echo "ERROR: Cannot determine repository root from SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
    exit 1
fi

CONFIG_FILE="$REPO_ROOT/config.yaml"

# Check if config.yaml exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config.yaml not found at: $CONFIG_FILE"
    echo "Please copy config.yaml.template to config.yaml and fill in your paths"
    exit 1
fi

# Parse config.yaml using Python (more reliable than bash parsing)
# This extracts key configuration values and exports them as environment variables
eval $(python3 -c "
import yaml
import sys
import os

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)

    # Python executable for SLURM
    slurm_python = config.get('python', {}).get('slurm_bin', 'env/bin/python')
    # Make it absolute if it's relative
    if not slurm_python.startswith('/'):
        slurm_python = os.path.join('$REPO_ROOT', slurm_python)
    print(f'export PYTHON_BIN=\"{slurm_python}\"')

    # Data path
    data_path = config.get('paths', {}).get('data', './data')
    if not data_path.startswith('/'):
        data_path = os.path.join('$REPO_ROOT', data_path)
    print(f'export DATA_PATH=\"{data_path}\"')

    # Output paths
    figures_path = config.get('paths', {}).get('figures', 'reports/figures')
    if not figures_path.startswith('/'):
        figures_path = os.path.join('$REPO_ROOT', figures_path)
    print(f'export FIGURES_PATH=\"{figures_path}\"')

    tables_path = config.get('paths', {}).get('tables', 'reports/tables')
    if not tables_path.startswith('/'):
        tables_path = os.path.join('$REPO_ROOT', tables_path)
    print(f'export TABLES_PATH=\"{tables_path}\"')

except Exception as e:
    print(f'echo \"ERROR parsing config.yaml: {e}\"', file=sys.stderr)
    print('exit 1', file=sys.stderr)
" 2>&1)

# Verify Python executable exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python executable not found at: $PYTHON_BIN"
    echo "Please check python.slurm_bin in config.yaml"
    exit 1
fi

# Export repo root for script paths
export SCRIPTS_DIR="$REPO_ROOT/shinobi_fmri"
export LOGS_DIR="$REPO_ROOT/logs"
