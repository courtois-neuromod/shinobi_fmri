# shinobi_fmri

fMRI analysis pipeline for the [cneuromod.shinobi](https://github.com/courtois-neuromod/shinobi_access) dataset.

## Installation

This package requires Python 3.8+.

### Quick Setup (Recommended)

Use the automated setup script:

```bash
# Clone the repository
git clone https://github.com/courtois-neuromod/shinobi_fmri.git
cd shinobi_fmri

# Run automated setup
./setup.sh
```

The setup script will:
- Check Python version requirements
- Create a virtual environment in `./env/`
- Install all dependencies with pinned versions
- Create `config.yaml` from template
- Verify installation

### Manual Installation

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

**For detailed documentation of all tasks, see [TASKS.md](TASKS.md).**

### Quick Start

List all available tasks:
```bash
invoke --list
```

Display configuration and environment info:
```bash
invoke info
```

### Available Tasks

The pipeline provides the following task categories:

**GLM Analysis:**
- `glm.run-level` - Run-level (first-level) GLM analysis
- `glm.session-level` - Session-level (second-level) GLM analysis
- `glm.subject-level` - Subject-level (third-level) GLM analysis

**MVPA:**
- `mvpa.session-level` - Multi-Voxel Pattern Analysis (classification/decoding)
- `mvpa.permutations` - Distributed permutation testing for significance
- `mvpa.aggregate-permutations` - Aggregate permutation results and compute p-values

**Correlation Analysis:**
- `corr.beta` - Compute beta map correlations with HCP data
- `corr.fingerprinting` - Subject identification from brain map similarity

**Visualization:**
- `viz.run-level` - Run-level visualizations
- `viz.session-level` - Session-level visualizations
- `viz.subject-level` - Subject-level visualizations
- `viz.annotation-panels` - Generate annotation panels and PDFs
- `viz.beta-correlations` - Generate beta correlations figure
- `viz.regressor-correlations` - Design matrix regressor correlation matrices
- `viz.condition-comparison` - Condition comparison surface plots
- `viz.atlas-tables` - Generate atlas tables for z-maps
- `viz.fingerprinting` - Fingerprinting analysis visualizations
- `viz.within-subject-correlations` - Within-subject condition correlation analysis and heatmaps
- `viz.mvpa-confusion-matrices` - MVPA confusion matrices with task grouping

**Pipelines:**
- `pipeline.full` - Complete pipeline for a subject/session
- `pipeline.subject` - Complete subject-level pipeline

### Example Usage

```bash
# Run GLM analysis for a subject/session
invoke glm.run-level --subject sub-01 --session ses-001 --verbose 1

# Submit all correlation chunks to SLURM
invoke corr.beta --slurm --chunk-size 100 --verbose 1

# Generate visualizations for all subjects
invoke viz.session-level

# Run complete pipeline
invoke pipeline.full --subject sub-01 --session ses-001
```

**For more examples and detailed argument documentation, see [TASKS.md](TASKS.md).**

## Reproducibility Features

This pipeline implements comprehensive provenance tracking for full reproducibility:

### Metadata JSON Sidecars
Every analysis output (z-maps, beta-maps) is automatically accompanied by a `.json` metadata file containing:
- **Git commit hash** - links results to exact code version
- **Analysis parameters** - all settings used (thresholds, models, etc.)
- **Software versions** - Python, nilearn, numpy, scipy, and all dependencies
- **Timestamps** - when the analysis was run
- **Subject/session info** - BIDS identifiers
- **Warnings** - flags uncommitted changes in repository

Example: When you generate `sub-01_ses-001_task-shinobi_run-01_contrast-HIT_stat-z.nii.gz`, the pipeline also creates `sub-01_ses-001_task-shinobi_run-01_contrast-HIT_stat-z.json` with full provenance information.

### Dataset Descriptions
Each processed output directory contains a BIDS-compliant `dataset_description.json` describing:
- Pipeline name and version
- Source data
- Key parameters
- Generation timestamp
- Git commit and branch

### Pinned Dependencies
All package versions are pinned in `requirements.txt` (21 packages with exact versions) ensuring consistent results across installations.

### How to Use for Publications
1. Run your analysis normally - metadata is saved automatically
2. Note the git commit hash from the log or metadata files
3. In your paper's methods section, cite the repository and commit hash
4. Readers can reproduce your exact results using that commit

See `AGENTS.md` for complete reproducible science guidelines.

## Project Structure

- `shinobi_fmri/`: Main package source code
  - `glm/`: General Linear Model analysis scripts
  - `mvpa/`: Multi-Voxel Pattern Analysis scripts
  - `correlations/`: Correlation analysis scripts
  - `visualization/`: Plotting and reporting tools
  - `utils/`: Shared utilities (logger, provenance tracking, etc.)