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

# Find config file at repo root (one level up from shinobi_fmri/)
_repo_root = op.dirname(op.dirname(op.abspath(__file__)))
_config_file = op.join(_repo_root, 'config.yaml')
_config_template = op.join(_repo_root, 'config.yaml.template')

# Load config (with helpful error if not found)
try:
    with open(_config_file, 'r') as f:
        _config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "\n" + "="*60 + "\n"
        "CONFIG FILE NOT FOUND\n"
        "="*60 + "\n"
        f"Expected: {_config_file}\n\n"
        "SETUP REQUIRED:\n"
        "  1. Copy the template:\n"
        f"     cp config.yaml.template config.yaml\n\n"
        "  2. Edit config.yaml and replace <PLACEHOLDER> values with your paths\n\n"
        "See README.md for detailed setup instructions.\n"
        "="*60
    )

# Check for placeholders still present
_config_str = str(_config)
if '<PATH' in _config_str or '<PLACEHOLDER>' in _config_str:
    raise ValueError(
        "\n" + "="*60 + "\n"
        "CONFIG NOT SETUP PROPERLY\n"
        "="*60 + "\n"
        "Your config.yaml still contains placeholder values like:\n"
        "  <PATH_TO_YOUR_DATA> or <PATH_TO_PYTHON>\n\n"
        "Please edit config.yaml and replace ALL placeholders\n"
        "with your actual paths.\n"
        "="*60
    )

# Expose paths
DATA_PATH = _config['paths']['data']
FIG_PATH = _config['paths']['figures']
TABLE_PATH = _config['paths']['tables']

# Expose analysis parameters
SUBJECTS = _config['analysis']['subjects']
CONDITIONS = _config['analysis']['conditions']
CONDS_LIST = CONDITIONS  # Alias for backward compatibility
LOW_LEVEL_CONDITIONS = _config['analysis']['low_level_conditions']

# Behavioral analysis parameters
LEVELS = _config['analysis']['levels']
ACTIONS = _config['analysis']['actions']
GAME_FS = _config['analysis']['game_fs']

# Python environments
PYTHON_BIN = _config['python']['local_bin']
SLURM_PYTHON_BIN = _config['python']['slurm_bin']

# GLM parameters
GLM_TR = _config['glm']['tr']
GLM_HRF_MODEL = _config['glm']['hrf_model']
GLM_SMOOTHING_FWHM = _config['glm']['smoothing_fwhm']
GLM_N_JOBS = _config['glm']['n_jobs']
GLM_ALPHA = _config['glm']['alpha']
GLM_CLUSTER_THRESH_RUN = _config['glm']['cluster_thresh_run']
GLM_CLUSTER_THRESH_SESSION = _config['glm']['cluster_thresh_session']
GLM_CLUSTER_THRESH_SUBJECT = _config['glm']['cluster_thresh_subject']
GLM_NOISE_MODEL = _config['glm']['noise_model']
GLM_DRIFT_MODEL = _config['glm']['drift_model']

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
