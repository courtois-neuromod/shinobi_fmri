"""
Shinobi fMRI Configuration Module

Loads configuration from config.yaml and exposes constants.

Setup:
  1. Copy config.yaml.template to config.yaml
  2. Fill in your local paths
  3. Import: from shinobi_fmri.config import DATA_PATH
"""

import os
import os.path as op
import yaml
from pathlib import Path

# Find config file
_config_file = op.join(op.dirname(op.abspath(__file__)), 'config.yaml')

# Load config (with helpful error if not found)
try:
    with open(_config_file, 'r') as f:
        _config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Config file not found: {_config_file}\n"
        f"Please copy config.yaml.template to config.yaml and fill in your paths:\n"
        f"  cp {op.join(op.dirname(_config_file), 'config.yaml.template')} {_config_file}"
    )

# Expose paths
DATA_PATH = _config['paths']['data']
FIG_PATH = _config['paths']['figures']
TABLE_PATH = _config['paths']['tables']

# Expose analysis parameters
SUBJECTS = _config['analysis']['subjects']
CONDITIONS = _config['analysis']['conditions']
CONDS_LIST = CONDITIONS  # Alias for backward compatibility

# Behavioral analysis parameters
LEVELS = _config['analysis']['levels']
ACTIONS = _config['analysis']['actions']
GAME_FS = _config['analysis']['game_fs']

# Python environments
PYTHON_BIN = _config['python']['local_bin']
SLURM_PYTHON_BIN = _config['python']['slurm_bin']

# Derived paths
PROJECT_ROOT = op.dirname(op.abspath(__file__))
SHINOBI_FMRI_DIR = op.join(PROJECT_ROOT, 'shinobi_fmri')
SLURM_DIR = op.join(PROJECT_ROOT, 'slurm')

# For lowercase aliases (used in some older scripts)
subjects = SUBJECTS
figures_path = FIG_PATH
path_to_data = DATA_PATH


def get_config():
    """Return the full config dictionary."""
    return _config


def get_path(key):
    """Get a path from config."""
    return _config['paths'].get(key)


# Validate that DATA_PATH exists
if not op.exists(DATA_PATH):
    import warnings
    warnings.warn(
        f"DATA_PATH does not exist: {DATA_PATH}\n"
        f"Please update config.yaml with the correct path.",
        UserWarning
    )
