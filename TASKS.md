# Shinobi fMRI Analysis Tasks

This document provides detailed documentation for all available analysis tasks in the shinobi_fmri pipeline. Tasks can be run using the `invoke` command.

## Table of Contents

- [Common Arguments](#common-arguments)
- [GLM Analysis Tasks](#glm-analysis-tasks)
- [MVPA Tasks](#mvpa-tasks)
- [Correlation Analysis Tasks](#correlation-analysis-tasks)
- [Visualization Tasks](#visualization-tasks)
- [Pipeline Tasks](#pipeline-tasks)
- [Setup and Utility Tasks](#setup-and-utility-tasks)

---

## Common Arguments

Most tasks support these common arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | int | 0 | Verbosity level (0=WARNING, 1=INFO, 2=DEBUG) |
| `--log-dir` | str | `./logs/` | Custom directory for log files |
| `--slurm` | flag | False | Submit job to SLURM cluster |
| `--n-jobs` | int | -1 | Number of parallel jobs (-1 = all CPU cores) |

---

## GLM Analysis Tasks

### `glm.run-level`

Run run-level (first-level) GLM analysis for individual fMRI runs.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, process all subjects |
| `--session` | str | None | No | Session ID (e.g., `ses-001`). If None, process all sessions |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |
| `--low-level-confs` | flag | False | No | Include low-level confounds and button-press rate in design matrix |

**Common Use Cases:**

```bash
# Process a single subject/session locally
invoke glm.run-level --subject sub-01 --session ses-001 --verbose 1

# Process a single subject/session on SLURM
invoke glm.run-level --subject sub-01 --session ses-001 --slurm

# Process all sessions for a specific subject
invoke glm.run-level --subject sub-01

# Process all subjects and sessions locally
invoke glm.run-level

# Process all with SLURM batch submission
invoke glm.run-level --slurm

# Include low-level confounds in the design matrix
invoke glm.run-level --subject sub-01 --session ses-001 --low-level-confs
```

---

### `glm.session-level`

Run session-level (second-level) GLM analysis by combining runs within a session.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, process all subjects |
| `--session` | str | None | No | Session ID (e.g., `ses-001`). If None, process all sessions |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |
| `--low-level-confs` | flag | False | No | Include low-level confounds and button-press rate in design matrix |

**Common Use Cases:**

```bash
# Process a single subject/session
invoke glm.session-level --subject sub-01 --session ses-001 --verbose 1

# Process all sessions for a subject
invoke glm.session-level --subject sub-01

# Process all subjects/sessions with SLURM
invoke glm.session-level --slurm

# Include low-level confounds
invoke glm.session-level --subject sub-01 --session ses-001 --low-level-confs
```

---

### `glm.subject-level`

Run subject-level (third-level) GLM analysis by combining sessions across a subject.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, process all subjects |
| `--condition` | str | None | No | Condition/contrast name (e.g., `HIT`, `JUMP`, `Kill`). If None, process all conditions |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Process a single subject/condition
invoke glm.subject-level --subject sub-01 --condition HIT --verbose 1

# Process all conditions for a subject
invoke glm.subject-level --subject sub-01

# Process all subjects/conditions with SLURM
invoke glm.subject-level --slurm

# Process specific condition across all subjects
invoke glm.subject-level --condition JUMP
```

---

## MVPA Tasks

### `mvpa.session-level`

Run complete session-level Multi-Voxel Pattern Analysis (MVPA) pipeline: decoder + permutation testing + aggregation. Supports both local (sequential) and SLURM (parallel with job dependencies) execution modes.

The task automatically:
1. Fits decoder and computes cross-validated accuracies, confusion matrices
2. Runs permutation testing for statistical significance (optional)
3. Aggregates permutation results and computes p-values (optional)

**Key Features:**
- **SLURM mode**: Uses job dependencies to automatically chain decoder → permutations → aggregation
- **Local mode**: Runs all steps sequentially
- **Flexible**: Skip any step (decoder, permutations, aggregation) as needed

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, processes all subjects |
| `--n-permutations` | int | 1000 | No | Total number of permutations (set to 0 to skip permutation testing) |
| `--perms-per-job` | int | 50 | No | Number of permutations per SLURM job (ignored for local mode) |
| `--screening` | int | 20 | No | Feature screening percentile (1-100) |
| `--n-jobs` | int | -1 | No | Number of parallel jobs for decoder fitting |
| `--slurm` | flag | False | No | Submit to SLURM cluster with job dependencies |
| `--skip-decoder` | flag | False | No | Skip decoder step (run only permutations/aggregation) |
| `--skip-permutations` | flag | False | No | Skip permutation testing |
| `--skip-aggregate` | flag | False | No | Skip aggregation step |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Run complete pipeline locally for one subject (decoder + 1000 permutations + aggregation)
invoke mvpa.session-level --subject sub-01

# Run complete pipeline on SLURM for all subjects
# Automatically chains jobs: decoder → permutations → aggregation with dependencies
invoke mvpa.session-level --slurm

# Run only decoder (no permutations)
invoke mvpa.session-level --subject sub-01 --n-permutations 0

# Quick test with fewer permutations
invoke mvpa.session-level --subject sub-01 --n-permutations 100 --perms-per-job 10

# SLURM: Run only permutations and aggregation (decoder already completed)
invoke mvpa.session-level --subject sub-01 --skip-decoder --slurm

# Local: Run only aggregation (decoder and permutations already done)
invoke mvpa.session-level --subject sub-01 --skip-decoder --skip-permutations

# Custom screening with full pipeline on SLURM
invoke mvpa.session-level --screening 10 --n-permutations 1000 --slurm
```

**How Job Dependencies Work (SLURM Mode):**

When you run with `--slurm`, the task:
1. Submits decoder job → captures job ID
2. Submits permutation jobs (e.g., 20 jobs for 1000 perms) → captures all job IDs
3. Submits aggregation job with `--dependency=afterok:decoder_id:perm_id1:perm_id2:...`

The aggregation job waits for ALL previous jobs (both decoder and all permutations) to complete successfully before running.

**Complete Workflow Example:**

```bash
# Single command runs everything on SLURM with proper job dependencies
invoke mvpa.session-level --subject sub-01 --n-permutations 1000 --slurm

# Monitor jobs
squeue -u $USER

# After completion, visualize results
invoke viz.mvpa-confusion-matrices
```

---

## Correlation Analysis Tasks

### `corr.beta`

Compute beta map correlation matrices with HCP data.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--chunk-start` | int | 0 | No | Starting index for chunked processing (local mode only) |
| `--chunk-size` | int | 100 | No | Number of maps per chunk |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |
| `--slurm` | flag | False | No | Automatically submit all chunks as SLURM jobs |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Submit all chunks to SLURM (recommended for large datasets)
invoke corr.beta --slurm --chunk-size 100 --verbose 1

# Run single chunk locally (for testing)
invoke corr.beta --chunk-start 0 --chunk-size 100 --n-jobs 20 --verbose 1

# Process larger chunks
invoke corr.beta --chunk-start 200 --chunk-size 200 --n-jobs 32
```

**How it works:**
- When `--slurm` is used, the script automatically discovers all beta maps, splits them into chunks, and submits one SLURM job per chunk
- Each job computes correlations only for missing pairs in that chunk
- Useful for processing large correlation matrices in parallel

---

### `corr.fingerprinting`

Run fingerprinting analysis on beta maps to assess participant-specificity.

**Description:**

Assesses whether brain maps are participant-specific by checking if each map's most similar map (nearest neighbor) comes from the same subject. This provides a quantitative measure of individual differences in brain activation patterns.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**
- `processed/fingerprinting/fingerprinting_detailed.tsv` - Per-map results
- `processed/fingerprinting/fingerprinting_aggregated.tsv` - Summary statistics

**Common Use Cases:**

```bash
# Run fingerprinting analysis
invoke corr.fingerprinting --verbose 1

# With custom log directory
invoke corr.fingerprinting --log-dir ./logs/fingerprinting
```

**What it computes:**
- For each map: finds nearest neighbor (highest correlation) and checks if from same subject
- Fingerprinting score = proportion of maps where nearest neighbor is from same subject
- Aggregates by subject, condition, and analysis level

---

### `corr.within-subject-conditions`

Compute within-subject condition correlations to assess condition specificity.

**Description:**

For each subject, computes how maps from different conditions correlate with each other. This reveals how distinct different experimental conditions are within each individual.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**
- `processed/within_subject_condition_correlations/{subject}_condition_correlations.tsv` - Per-subject condition correlation matrices
- `processed/within_subject_condition_correlations/average_condition_correlations.tsv` - Average across subjects

**Common Use Cases:**

```bash
# Compute within-subject condition correlations
invoke corr.within-subject-conditions --verbose 1

# With custom log directory
invoke corr.within-subject-conditions --log-dir ./logs/condition_corr
```

**What it computes:**
- Condition × condition correlation matrix for each subject
- Average correlation matrices across all subjects
- Same-condition vs different-condition correlation statistics

---

## Visualization Tasks

### `viz.run-level`

Generate visualizations for run-level GLM results.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | - | Yes | Subject ID (e.g., `sub-01`) |
| `--condition` | str | - | Yes | Condition/contrast name |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate run-level visualizations
invoke viz.run-level --subject sub-01 --condition HIT --verbose 1

# Submit to SLURM
invoke viz.run-level --subject sub-01 --condition JUMP --slurm
```

---

### `viz.session-level`

Generate visualizations for session-level GLM results.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, process all |
| `--condition` | str | None | No | Condition/contrast name. If None, process all |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate session-level visualizations for specific subject/condition
invoke viz.session-level --subject sub-01 --condition HIT --verbose 1

# Process all subjects/conditions
invoke viz.session-level

# Submit to SLURM
invoke viz.session-level --slurm
```

---

### `viz.subject-level`

Generate visualizations for subject-level GLM results.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate subject-level visualizations
invoke viz.subject-level --verbose 1

# Submit to SLURM
invoke viz.subject-level --slurm
```

---

### `viz.annotation-panels`

Generate annotation panels with subject-level and session-level brain maps for documentation.

**Creates:**
- Individual inflated brain maps for each subject/session
- Combined panels (1 subject-level + top 4 session-level maps per subject)
- PDF with all annotation panels

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--condition` | str | None | No | Single condition to process (e.g., `HIT`) |
| `--conditions` | str | None | No | Comma-separated conditions (e.g., `HIT,JUMP,Kill`) |
| `--skip-individual` | flag | False | No | Skip generating individual brain maps |
| `--skip-panels` | flag | False | No | Skip generating annotation panels |
| `--skip-pdf` | flag | False | No | Skip generating PDF |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate all annotation panels for a single condition
invoke viz.annotation-panels --condition HIT --verbose 1

# Generate for multiple conditions
invoke viz.annotation-panels --conditions HIT,JUMP,Kill

# Skip individual maps, only create panels and PDF
invoke viz.annotation-panels --condition HIT --skip-individual

# Only generate individual maps (no panels or PDF)
invoke viz.annotation-panels --condition HIT --skip-panels --skip-pdf

# Process all default conditions
invoke viz.annotation-panels
```

---

### `viz.beta-correlations`

Generate beta correlations figure.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--input-path` | str | `{DATA_PATH}/processed/beta_correlations.pkl` | No | Path to input .pkl file |
| `--output-path` | str | `{FIG_PATH}/beta_correlations_plot.png` | No | Path to save output figure |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate figure with default paths
invoke viz.beta-correlations --verbose 1

# Use custom input/output paths
invoke viz.beta-correlations --input-path /path/to/correlations.pkl --output-path /path/to/output.png
```

---

### `viz.regressor-correlations`

Generate correlation matrices for design matrix regressors.

Creates correlation heatmaps and clustermaps showing relationships between all regressors (annotations) in the GLM design matrices. Produces both per-run and subject-averaged visualizations.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Specific subject to process. If None, process all subjects |
| `--skip-generation` | flag | False | No | Skip design matrix generation, only plot from existing pickle |
| `--low-level-confs` | flag | False | No | Include low-level confounds (psychophysics and button presses) |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate regressor correlation matrices for all subjects
invoke viz.regressor-correlations --verbose 1

# Process specific subject
invoke viz.regressor-correlations --subject sub-01

# Include low-level confounds
invoke viz.regressor-correlations --low-level-confs

# Only plot from existing pickle (skip regeneration)
invoke viz.regressor-correlations --skip-generation
```

---

### `viz.condition-comparison`

Generate condition comparison surface plots.

Creates surface plots comparing two conditions with three-color overlay:
- **Blue**: Significant only for condition 1
- **Red**: Significant only for condition 2
- **Purple**: Significant for both conditions

**Predefined comparisons (when using `--run-all`):**
- Kill vs reward (shinobi vs HCP gambling)
- HealthLoss vs punishment (shinobi vs HCP gambling)
- RIGHT vs JUMP (shinobi vs shinobi)
- LEFT vs HIT (shinobi vs shinobi)

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--cond1` | str | None | No | First condition in format `source:condition` (e.g., `shinobi:Kill` or `hcp:reward`) |
| `--cond2` | str | None | No | Second condition in format `source:condition` (e.g., `shinobi:HealthLoss` or `hcp:punishment`) |
| `--run-all` | flag | False | No | Generate all predefined comparisons (default if no conditions specified) |
| `--threshold` | float | 3.0 | No | Significance threshold for z-maps |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |
| `--output-dir` | str | `reports/figures/condition_comparison/` | No | Custom output directory |

**Common Use Cases:**

```bash
# Generate all predefined comparisons
invoke viz.condition-comparison --run-all --verbose 1

# Compare two specific conditions
invoke viz.condition-comparison --cond1 shinobi:Kill --cond2 hcp:reward

# Use custom threshold
invoke viz.condition-comparison --run-all --threshold 2.5

# Custom output directory
invoke viz.condition-comparison --cond1 shinobi:LEFT --cond2 shinobi:RIGHT --output-dir /custom/path
```

---

### `viz.atlas-tables`

Generate atlas tables for z-maps, identifying significant clusters and their anatomical locations.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--input-dir` | str | `{DATA_PATH}/processed/z_maps/subject-level` | No | Directory containing z-maps |
| `--output-dir` | str | `reports/tables` | No | Directory to save output tables |
| `--cluster-extent` | int | 5 | No | Minimum cluster size in voxels |
| `--voxel-thresh` | float | 3.0 | No | Voxel threshold for significance |
| `--direction` | str | `both` | No | Direction of the contrast (`both`, `pos`, `neg`) |
| `--overwrite` | flag | False | No | Overwrite existing cluster files |

**Common Use Cases:**

```bash
# Generate atlas tables with default settings
invoke viz.atlas-tables

# Use custom thresholds
invoke viz.atlas-tables --cluster-extent 10 --voxel-thresh 2.5

# Only positive activations
invoke viz.atlas-tables --direction pos

# Use custom input/output directories
invoke viz.atlas-tables --input-dir /path/to/zmaps --output-dir /path/to/tables

# Force overwrite existing files
invoke viz.atlas-tables --overwrite
```

---

### `viz.fingerprinting`

Generate fingerprinting analysis visualizations.

**Description:**

Creates comprehensive visualizations showing subject identification from brain map similarity, including confusion matrices, correlation distributions, and fingerprinting scores by different groupings.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**
- `reports/figures/fingerprinting/fingerprinting_confusion_matrix.png` - Subject identification confusion matrix
- `reports/figures/fingerprinting/fingerprinting_correlation_distributions.png` - Within vs between subject correlations
- `reports/figures/fingerprinting/fingerprinting_by_source.png` - Scores by analysis level
- `reports/figures/fingerprinting/fingerprinting_by_condition.png` - Scores by condition
- `reports/figures/fingerprinting/fingerprinting_nearest_neighbor_correlations.png` - Distribution of nearest neighbor similarities

**Common Use Cases:**

```bash
# Generate fingerprinting visualizations
invoke viz.fingerprinting --verbose 1

# With custom log directory
invoke viz.fingerprinting --log-dir ./logs/viz_fingerprinting
```

**What it generates:**
- Confusion matrix showing perfect diagonal (subject identification accuracy)
- Histograms and violin plots comparing within-subject vs between-subject correlations
- Bar plots showing fingerprinting performance across conditions and analysis levels
- Statistical summaries including Cohen's d and t-test results

---

### `viz.within-subject-conditions`

Generate within-subject condition correlation visualizations.

**Description:**

Creates heatmaps and plots showing how different experimental conditions correlate with each other within each subject, demonstrating condition specificity and distinctiveness.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**
- `reports/figures/within_subject_condition_correlations/{subject}_condition_correlations.png` - Per-subject condition correlation heatmaps
- `reports/figures/within_subject_condition_correlations/average_condition_correlations.png` - Average heatmap across subjects
- `reports/figures/within_subject_condition_correlations/same_vs_different_conditions.png` - Same vs different condition comparison
- `reports/figures/within_subject_condition_correlations/condition_specificity_matrix.png` - Specificity scores by condition and subject

**Common Use Cases:**

```bash
# Generate within-subject condition visualizations
invoke viz.within-subject-conditions --verbose 1

# With custom log directory
invoke viz.within-subject-conditions --log-dir ./logs/viz_conditions
```

**What it generates:**
- Condition × condition correlation heatmaps for each subject (separated by Shinobi/HCP)
- Violin and box plots comparing same-condition vs different-condition correlations
- Condition specificity matrix showing which conditions are most distinctive
- Statistical comparisons including Cohen's d and significance tests

---

### `viz.mvpa-confusion-matrices`

Generate publication-quality confusion matrix visualization for all subjects with task icons and separation styling.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--screening` | int | 20 | No | Screening percentile used |
| `--output` | str | Auto | No | Output path for figure |

**Output:**
- `reports/figures/mvpa_confusion_matrices_s{screening}.png` - 2x2 grid of confusion matrices

**Common Use Cases:**

```bash
# Generate figure with default settings
invoke viz.mvpa-confusion-matrices

# Custom screening percentile
invoke viz.mvpa-confusion-matrices --screening 10

# Custom output location
invoke viz.mvpa-confusion-matrices --output reports/my_figure.png
```

**What it generates:**
- 2×2 grid showing confusion matrices for all 4 subjects
- Task icons (♦ ✱ ■ ★ ● ▲) in tick labels for HCP conditions
- Color-coded labels by task
- Visual separation between Shinobi and HCP task blocks
- Shared colorbar and legend showing all tasks

---

## Pipeline Tasks

These tasks run complete analysis pipelines combining multiple steps.

### `pipeline.full`

Run complete analysis pipeline for a single subject/session.

**Pipeline stages:**
1. Run-level GLM
2. Session-level GLM
3. Visualizations

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | - | Yes | Subject ID (e.g., `sub-01`) |
| `--session` | str | - | Yes | Session ID (e.g., `ses-001`) |
| `--slurm` | flag | False | No | Submit each stage to SLURM |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |

**Common Use Cases:**

```bash
# Run full pipeline locally
invoke pipeline.full --subject sub-01 --session ses-001

# Run with SLURM
invoke pipeline.full --subject sub-01 --session ses-001 --slurm

# Use specific number of cores
invoke pipeline.full --subject sub-01 --session ses-001 --n-jobs 8
```

---

### `pipeline.subject`

Run complete subject-level analysis pipeline.

**Pipeline stages:**
1. Subject-level GLM for all conditions
2. Subject-level visualizations

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | - | Yes | Subject ID (e.g., `sub-01`) |
| `--slurm` | flag | False | No | Submit each stage to SLURM |
| `--n-jobs` | int | -1 | No | Number of parallel jobs |

**Common Use Cases:**

```bash
# Run subject pipeline locally
invoke pipeline.subject --subject sub-01

# Run with SLURM
invoke pipeline.subject --subject sub-01 --slurm

# Use specific number of cores
invoke pipeline.subject --subject sub-01 --n-jobs 16
```

---

## Setup and Utility Tasks

### `setup.env`

Install Python dependencies from requirements.txt.

**Arguments:** None

**Common Use Cases:**

```bash
# Install all dependencies
invoke setup.env
```

---

### `info`

Display configuration and environment information.

Shows current configuration including:
- Project paths
- Python binaries
- Subjects and conditions
- Available sessions and runs

**Arguments:** None

**Common Use Cases:**

```bash
# Display environment info
invoke info
```

---

## Additional Notes

### List All Available Tasks

```bash
invoke --list
```

### Batch Processing

Many tasks support batch processing by omitting subject/session/condition arguments:

```bash
# Process all subjects/sessions
invoke glm.run-level

# Process all conditions for a subject
invoke glm.subject-level --subject sub-01

# Process all subjects for a condition
invoke glm.subject-level --condition HIT
```

### SLURM Integration

When using `--slurm`, tasks are submitted to the SLURM cluster. The task will:
1. Use the appropriate SLURM submission script from the `slurm/` directory
2. Use the Python binary configured in `config.yaml` for SLURM
3. Return immediately after job submission (non-blocking)

### Logging

All tasks create detailed log files in `./logs/<module>/` by default. Use `--log-dir` to customize the location. Control verbosity with `--verbose`:
- `0` (default): Warnings and vital info
- `1` (`-v`): Progress tracking, success/skip messages
- `2` (`-vv`): Detailed steps, full tracebacks
