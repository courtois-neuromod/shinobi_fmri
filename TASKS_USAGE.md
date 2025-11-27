# Shinobi fMRI Analysis Pipeline - Task Automation Guide

This guide explains how to use the `tasks.py` automation system built with Airoh/Invoke for running your fMRI analysis pipeline both locally and on Compute Canada clusters.

## Quick Start

### 1. Installation

First, install the required dependencies:

```bash
# Install invoke and airoh
pip install invoke airoh

# Or install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Verify Installation

Check that everything is set up correctly:

```bash
# List all available tasks
invoke --list

# Show configuration info
invoke info
```

## Usage Patterns

The task system supports **two execution modes**:

1. **Local execution**: Runs Python scripts directly on your current machine
2. **SLURM execution**: Submits jobs to Compute Canada using SLURM

Use the `--slurm` flag to switch between modes.

## Task Organization

Tasks are organized into namespaced collections:

```
glm.*              # GLM analysis tasks
mvpa.*             # MVPA analysis tasks
corr.*             # Correlation analysis tasks
viz.*              # Visualization tasks
batch.*            # Batch processing for multiple subjects/sessions
pipeline.*         # Full pipeline workflows
setup.*            # Environment setup
```

## Common Usage Examples

### Single Subject/Session Analysis

#### Run-level GLM

```bash
# Run locally
invoke glm.run-level --subject sub-01 --session ses-001

# Submit to SLURM
invoke glm.run-level --subject sub-01 --session ses-001 --slurm
```

#### Session-level GLM

```bash
# Run locally
invoke glm.session-level --subject sub-01 --session ses-001

# Submit to SLURM
invoke glm.session-level --subject sub-01 --session ses-001 --slurm
```

#### Subject-level GLM

```bash
# Run locally
invoke glm.subject-level --subject sub-01 --condition HIT

# Submit to SLURM
invoke glm.subject-level --subject sub-01 --condition HIT --slurm
```

### Batch Processing

#### Process All Run-level GLMs

```bash
# Run locally for all subjects/sessions
invoke batch.glm-run-level

# Submit all to SLURM
invoke batch.glm-run-level --slurm

# Process only one subject
invoke batch.glm-run-level --subject sub-01
invoke batch.glm-run-level --subject sub-01 --slurm
```

#### Process All Session-level GLMs

```bash
# Run locally for all subjects/sessions
invoke batch.glm-session-level

# Submit all to SLURM
invoke batch.glm-session-level --slurm

# Process only one subject
invoke batch.glm-session-level --subject sub-02
```

#### Process All Subject-level GLMs

```bash
# Run all subjects and conditions locally
invoke batch.glm-subject-level

# Submit all to SLURM
invoke batch.glm-subject-level --slurm

# Process specific subjects
invoke batch.glm-subject-level --subjects "sub-01,sub-02" --slurm

# Process specific conditions
invoke batch.glm-subject-level --conditions "HIT,JUMP,Kill" --slurm

# Combine both
invoke batch.glm-subject-level --subjects "sub-01" --conditions "HIT,JUMP" --slurm
```

### MVPA Analysis

```bash
# Run MVPA for a session
invoke mvpa.session-level --subject sub-01 --task shinobi --perm-index 0

# Submit to SLURM
invoke mvpa.session-level --subject sub-01 --task shinobi --perm-index 0 --slurm
```

### Correlation Analysis

```bash
# Run correlation analysis with chunking
invoke corr.compute --chunk-start 0 --chunk-size 100 --n-jobs 40

# Submit to SLURM
invoke corr.compute --chunk-start 0 --slurm

# Submit multiple correlation jobs (100 chunks of 100 maps each)
invoke batch.correlations --num-jobs 100 --chunk-size 100 --slurm
```

### Visualizations

```bash
# Run-level visualizations
invoke viz.run-level --subject sub-01 --condition HIT
invoke viz.run-level --subject sub-01 --condition HIT --slurm

# Session-level visualizations
invoke viz.session-level
invoke viz.session-level --slurm

# Subject-level visualizations
invoke viz.subject-level
invoke viz.subject-level --slurm

# Annotation panels (subject-level + top 4 session-level maps)
invoke viz.annotation-panels                                   # All conditions
invoke viz.annotation-panels --condition HIT                   # Single condition
invoke viz.annotation-panels --conditions "HIT,JUMP,Kill"      # Multiple conditions
invoke viz.annotation-panels --skip-pdf                        # Skip PDF generation
```

### Full Pipeline Workflows

#### Complete Pipeline for One Subject/Session

This runs the full pipeline: run-level → session-level → visualizations

```bash
# Run locally
invoke pipeline.full --subject sub-01 --session ses-001

# Submit to SLURM
invoke pipeline.full --subject sub-01 --session ses-001 --slurm
```

#### Complete Subject-level Pipeline

This runs subject-level GLM for all conditions + visualizations

```bash
# Run locally
invoke pipeline.subject --subject sub-01

# Submit to SLURM
invoke pipeline.subject --subject sub-01 --slurm
```

## Environment Setup

### Install Dependencies

```bash
# Install all Python dependencies
invoke setup.env

# Install airoh and invoke specifically
invoke setup.airoh
```

### Check Configuration

```bash
# Display current configuration and available data
invoke info
```

## Advanced Usage

### Chaining Multiple Commands

You can run multiple tasks sequentially:

```bash
# Run multiple tasks in order
invoke glm.run-level --subject sub-01 --session ses-001 && \
invoke glm.session-level --subject sub-01 --session ses-001 && \
invoke viz.session-level
```

### Custom Bash Scripts

You can still use your existing bash scripts if needed, or integrate them into custom tasks in `tasks.py`.

### Monitoring SLURM Jobs

After submitting jobs to SLURM, monitor them with standard SLURM commands:

```bash
# Check job queue
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Typical Workflows

### Workflow 1: Process New Subject Data

When you have new data for a subject:

```bash
# 1. Process all runs for the subject (submit to SLURM)
invoke batch.glm-run-level --subject sub-06 --slurm

# 2. Wait for jobs to complete, then process sessions
invoke batch.glm-session-level --subject sub-06 --slurm

# 3. Wait for jobs to complete, then process subject level
invoke pipeline.subject --subject sub-06 --slurm
```

### Workflow 2: Reprocess Everything

To reprocess all subjects from scratch:

```bash
# Submit all run-level analyses
invoke batch.glm-run-level --slurm

# After completion, submit all session-level analyses
invoke batch.glm-session-level --slurm

# After completion, submit all subject-level analyses
invoke batch.glm-subject-level --slurm

# Finally, generate visualizations
invoke viz.subject-level --slurm
```

### Workflow 3: Quick Local Testing

Before submitting large batches to SLURM, test locally:

```bash
# Test on one subject/session locally
invoke glm.run-level --subject sub-01 --session ses-001

# If it works, submit the full batch
invoke batch.glm-run-level --slurm
```

### Workflow 4: Correlation Analysis

For large-scale correlation analysis with chunking:

```bash
# Submit 100 jobs, each processing 100 maps
invoke batch.correlations --num-jobs 100 --chunk-size 100 --slurm

# Monitor progress
squeue -u $USER | grep shi_corr_chunk
```

## Advantages Over Manual Scripts

### Before (Manual SLURM Scripts)

```bash
# Had to navigate to slurm directory
cd slurm

# Had to remember exact script names and arguments
sbatch subm_run-level.sh sub-01 ses-001

# Had to manually loop for batch processing
for sub in sub-01 sub-02 sub-04 sub-06; do
    sbatch subm_subject-level.sh $sub HIT
done
```

### After (Tasks)

```bash
# Single command from anywhere in the project
invoke glm.run-level --subject sub-01 --session ses-001 --slurm

# Built-in batch processing
invoke batch.glm-subject-level --slurm
```

### Key Benefits

1. **Single Interface**: One command system for both local and cluster execution
2. **Discoverability**: `invoke --list` shows all available tasks
3. **Documentation**: `invoke --help <task>` shows task details
4. **Less Error-Prone**: No need to remember script paths or argument orders
5. **Flexible**: Easy to add new tasks or modify existing ones
6. **Version Controlled**: tasks.py is tracked in git
7. **Automation**: Built-in batch processing for common workflows

## Customization

### Modify Configuration

Edit the configuration section at the top of `tasks.py`:

```python
# Python environment for local execution
PYTHON_BIN = "python"  # Change if needed

# Python environment for SLURM execution
SLURM_PYTHON_BIN = "/home/hyruuk/python_envs/shinobi/bin/python"

# Analysis subjects and conditions
SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
CONDITIONS = ['HIT', 'JUMP', 'DOWN', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP']
```

### Add Custom Tasks

Add new tasks to `tasks.py`:

```python
@task
def my_custom_analysis(c, subject, slurm=False):
    """Run my custom analysis."""
    script = op.join(SHINOBI_FMRI_DIR, "custom", "my_script.py")

    if slurm:
        cmd = f"sbatch my_slurm_script.sh {subject}"
    else:
        cmd = f"{PYTHON_BIN} {script} --subject {subject}"

    c.run(cmd)

# Add to namespace
namespace.add_task(my_custom_analysis)
```

## Troubleshooting

### Task not found

```bash
# Make sure you're in the project root directory
cd /home/hyruuk/GitHub/neuromod/shinobi_fmri

# Verify tasks.py exists
ls -la tasks.py

# List available tasks
invoke --list
```

### Import errors

```bash
# Make sure dependencies are installed
pip install invoke airoh

# Check Python environment
which python
pip list | grep -E "(invoke|airoh)"
```

### Data path issues

```bash
# Check configuration
invoke info

# Verify data path in tasks.py matches your setup
# Edit DATA_PATH in tasks.py if needed
```

## Getting Help

```bash
# List all tasks
invoke --list

# Show detailed help for a specific task
invoke --help glm.run-level

# Show configuration and environment info
invoke info
```

## Additional Resources

- [Invoke documentation](https://www.pyinvoke.org/)
- [Airoh documentation](https://github.com/simexp/airoh)
- [Compute Canada SLURM guide](https://docs.alliancecan.ca/wiki/Running_jobs)
