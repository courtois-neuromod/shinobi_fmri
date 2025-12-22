# shinobi_fmri

fMRI analysis pipeline for the [cneuromod.shinobi](https://github.com/courtois-neuromod/shinobi_access) dataset.

## Installation

This package requires Python 3.8+.

```bash
# Clone the repository
git clone https://github.com/courtois-neuromod/shinobi_fmri.git
cd shinobi_fmri

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate

# Install the package and dependencies
pip install -e .
```

## Configuration

**First-time setup (required):**

1. Copy the configuration template:
   ```bash
   cp config.yaml.template config.yaml
   ```

2. Edit `config.yaml` and replace all `<PLACEHOLDER>` values with your actual paths:
   ```yaml
   paths:
     data: /your/path/to/shinobi_data  # Replace <PATH_TO_YOUR_DATA>
   python:
     local_bin: python                  # Or /path/to/your/env/bin/python
     slurm_bin: python                  # Or path to Python on HPC
   ```

3. The `config.yaml` file is gitignored - your paths stay private and local to your machine

**Note:** Each user/machine needs their own `config.yaml`. The file is not tracked by git, so everyone sets up their own paths.

## Logging System

All scripts come with a built-in logging system enabled by default.

- **Log Location**: Logs are saved to `./logs/<module>/` by default. You can override this with `--log-dir`.
- **Console Output**: Summaries and progress are shown in the console. 
- **Verbosity**: Control detail level with `-v` flags.
  - Default: Warnings and vital info.
  - `-v` (INFO): Progress tracking, success/skip messages.
  - `-vv` (DEBUG): Detailed steps, full tracebacks.

**Example Log Output:**
```
============================================================
Logging initialized for GLM_session
Subject: sub-01
Session: ses-001
Log file: ./logs/GLM_session/sub-01_ses-001_20250127_143022.log
============================================================
...
PROCESSING SUMMARY
============================================================
✓ Computed:  24
⊘ Skipped:   12
⚠ Warnings:  0
✗ Errors:    4
```

## Usage

You can run analysis tasks using `invoke` (recommended) or directly via python calls.

### Common Arguments for Invoke
- `--verbose`: Enable verbose output (0=WARNING, 1=INFO, 2=DEBUG)
- `--log-dir`: Custom directory for log files

### 1. GLM Analysis

**Run-level GLM:**
```bash
# Using invoke (recommended)
invoke glm.run-level --subject sub-01 --session ses-001 --verbose 1

# Direct execution
python -m shinobi_fmri.glm.compute_run_level -s sub-01 -ses ses-001 -v
```

**Session-level GLM:**
```bash
# Using invoke
invoke glm.session-level --subject sub-01 --session ses-001 --verbose 1

# Direct execution
python -m shinobi_fmri.glm.compute_session_level -s sub-01 -ses ses-001 -v
```

**Subject-level GLM:**
```bash
# Using invoke
invoke glm.subject-level --subject sub-01 --condition HIT --verbose 1

# Direct execution
python -m shinobi_fmri.glm.compute_subject_level -s sub-01 -cond HIT -v
```

### 2. MVPA

Run classification analysis (searchlight/decoding):

```bash
# Using invoke
invoke mvpa.session-level --subject sub-01 --verbose 1

# Direct execution
python -m shinobi_fmri.mvpa.compute_mvpa -s sub-01 --screening 20 -v
```

### 3. Correlation Analysis

Compute beta map correlation matrices:

```bash
# Submit all chunks to SLURM (recommended for large datasets)
invoke corr.beta --slurm --chunk-size 100 --verbose 1

# Run single chunk locally
invoke corr.beta --chunk-start 0 --chunk-size 100 --n-jobs 20 --verbose 1

# Direct execution with SLURM batch submission
python -m shinobi_fmri.correlations.compute_beta_correlations --slurm -v

# Direct execution for single chunk
python -m shinobi_fmri.correlations.compute_beta_correlations --chunk-start 0 --chunk-size 100 --n-jobs 20 -v
```

The `--slurm` flag automatically:
- Discovers all available beta maps
- Splits them into chunks (default: 100 maps per chunk)
- Submits one SLURM job per chunk
- Each job computes correlations only for missing pairs in that chunk

### 4. Visualization

Generate visualization reports:

```bash
# Using invoke
invoke viz.session-level --verbose 1

# Direct execution
python -m shinobi_fmri.visualization.viz_session-level -s sub-01 -c HIT -v
```

## Project Structure

- `shinobi_fmri/`: Main package source code
  - `glm/`: General Linear Model analysis scripts
  - `mvpa/`: Multi-Voxel Pattern Analysis scripts
  - `correlations/`: Correlation analysis scripts
  - `visualization/`: Plotting and reporting tools
  - `utils/`: Shared utilities (logger, etc.)