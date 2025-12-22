"""
Shinobi fMRI Configuration Module

Loads configuration from config.yaml and exposes constants for backward compatibility.
Environment variables can override certain config values.

Usage:
  Local: (default)
    from shinobi_fmri.config import DATA_PATH

  HPC: Set environment variable before importing
    export SHINOBI_ENV=hpc
    from shinobi_fmri.config import DATA_PATH
"""

import os
import os.path as op
import yaml
from pathlib import Path

# Find config file
_config_file = op.join(op.dirname(op.abspath(__file__)), 'config.yaml')

# Load config
with open(_config_file, 'r') as f:
    _config = yaml.safe_load(f)

# Determine environment (local, hpc, etc.)
SHINOBI_ENV = os.getenv('SHINOBI_ENV', 'local')

# Apply environment-specific overrides
if SHINOBI_ENV in _config.get('environments', {}):
    env_config = _config['environments'][SHINOBI_ENV]
    # Merge environment-specific paths
    if 'paths' in env_config:
        _config['paths'].update(env_config['paths'])
    # Merge environment-specific python settings
    if 'python' in env_config:
        _config['python'].update(env_config['python'])

# Expose paths (with environment variable overrides taking precedence)
DATA_PATH = os.getenv('SHINOBI_DATA_PATH', _config['paths']['data'])
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

# Python environments (with environment variable overrides taking precedence)
PYTHON_BIN = os.getenv('SHINOBI_PYTHON_BIN', _config['python']['local_bin'])
SLURM_PYTHON_BIN = os.getenv('SHINOBI_SLURM_PYTHON_BIN', _config['python']['slurm_bin'])

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
    """Get a path from config with environment variable override support."""
    env_var = f"SHINOBI_{key.upper()}_PATH"
    return os.getenv(env_var, _config['paths'].get(key))


# Validate that DATA_PATH exists
if not op.exists(DATA_PATH):
    import warnings
    warnings.warn(
        f"DATA_PATH does not exist: {DATA_PATH}\n"
        f"Set SHINOBI_DATA_PATH environment variable or update config.yaml",
        UserWarning
    )
