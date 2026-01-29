# Shinobi fMRI Analysis Tasks

This document provides detailed documentation for all available analysis tasks in the shinobi_fmri pipeline. Tasks can be run using the `invoke` command.

## Table of Contents

- [Common Arguments](#common-arguments)
- [GLM Analysis Tasks](#glm-analysis-tasks)
- [MVPA Tasks](#mvpa-tasks)
- [Correlation Analysis Tasks](#correlation-analysis-tasks)
- [Visualization Tasks](#visualization-tasks)
- [Descriptive Statistics Tasks](#descriptive-statistics-tasks)
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

**Common Use Cases:**

```bash
# Process a single subject/session
invoke glm.session-level --subject sub-01 --session ses-001 --verbose 1

# Process all sessions for a subject
invoke glm.session-level --subject sub-01

# Process all subjects/sessions with SLURM
invoke glm.session-level --slurm
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

### `glm.apply-cluster-correction`

Apply cluster-level FWE (family-wise error) correction to existing z-maps.

This task takes raw uncorrected z-maps and applies cluster-level correction, saving **properly thresholded z-maps** (not p-value maps) where only voxels in FWE-significant clusters retain their original z-values.

**Use this task to:**
- Re-run correction with different thresholds or alpha levels
- Fix existing p-value "corrected" maps to proper thresholded z-maps
- Apply correction to z-maps generated before this fix

**Statistical Method:**
- Uses nilearn's `cluster_level_inference` for cluster-level FWE correction
- Cluster-forming threshold: configurable (default: 2.3 for all levels)
- Family-wise error rate: alpha (default: 0.05)
- Only clusters with p < alpha survive correction

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--level` | str | `both` | Yes | Analysis level to process (`subject`, `session`, or `both`) |
| `--subject` | str | None | No | Process specific subject only (default: all subjects) |
| `--session` | str | None | No | Process specific session only (session-level only, default: all sessions) |
| `--threshold` | float | config | No | Cluster-forming threshold (default: 2.3 from config) |
| `--alpha` | float | 0.05 | No | Family-wise error rate |
| `--overwrite` | flag | False | No | Overwrite existing corrected z-maps |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Apply correction to all subject-level and session-level z-maps
invoke glm.apply-cluster-correction --level both -v

# Re-run subject-level correction with lower threshold (more liberal)
invoke glm.apply-cluster-correction --level subject --threshold 2.0 --overwrite -v

# Apply correction to specific subject's session-level maps
invoke glm.apply-cluster-correction --level session --subject sub-01 -v

# Use less stringent alpha for exploratory analysis
invoke glm.apply-cluster-correction --level both --alpha 0.10 -v

# Overwrite existing corrected maps (e.g., after changing config thresholds)
invoke glm.apply-cluster-correction --level both --overwrite -vv
```

**Notes:**
- Output files: `*_desc-corrected_stat-z.nii.gz`
- Contains original z-values for significant clusters, zero elsewhere
- Default threshold changed from 3.1 to 2.3 for subject-level (FSL's default)
- Appropriate for naturalistic paradigms with modest sample sizes

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
| `--exclude-low-level` | flag | False | No | Exclude low-level features from MVPA analysis |
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

# Exclude low-level features instead of including them
invoke mvpa.session-level --subject sub-01 --exclude-low-level --slurm
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
| `--exclude-low-level` | flag | False | No | Exclude low-level features from correlation matrix |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Submit all chunks to SLURM (recommended for large datasets)
invoke corr.beta --slurm --chunk-size 100 --verbose 1

# Exclude low-level features from correlations
invoke corr.beta --slurm --exclude-low-level --verbose 1

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

## Visualization Tasks

### `viz.session-level`

Generate visualizations for session-level GLM results.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Subject ID (e.g., `sub-01`). If None, process all |
| `--condition` | str | None | No | Condition/contrast name. If None, process all |
| `--slurm` | flag | False | No | Submit to SLURM cluster |
| `--exclude-low-level` | flag | False | No | Exclude low-level features from visualization |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate session-level visualizations for specific subject/condition
invoke viz.session-level --subject sub-01 --condition HIT --verbose 1

# Process all subjects/conditions
invoke viz.session-level

# Exclude low-level features from visualization
invoke viz.session-level --exclude-low-level

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
| `--exclude-low-level` | flag | False | No | Exclude low-level features from visualization |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Common Use Cases:**

```bash
# Generate subject-level visualizations
invoke viz.subject-level --verbose 1

# Exclude low-level features from visualization
invoke viz.subject-level --exclude-low-level

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
| `--use-corrected-maps` | flag | False | No | Use corrected z-maps instead of raw maps (default: raw maps) |
| `--exclude-low-level` | flag | False | No | Exclude low-level features from panels |
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

# Use corrected z-maps instead of raw maps
invoke viz.annotation-panels --condition HIT --use-corrected-maps

# Exclude low-level features from panels
invoke viz.annotation-panels --condition HIT --exclude-low-level

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

Creates correlation heatmaps and clustermaps showing relationships between all regressors (annotations) in the GLM design matrices. Produces both per-run and subject-averaged visualizations. The 2x2 subject-averaged grid includes Shinobi task conditions and low-level confounds (psychophysics: luminance, optical_flow, audio_envelope; button presses: button_presses_count).

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

# Exclude low-level features
invoke viz.regressor-correlations --exclude-low-level

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

*Original comparisons (4):*
- Kill vs reward (shinobi vs HCP gambling)
- HealthLoss vs punishment (shinobi vs HCP gambling)
- RIGHT vs JUMP (shinobi vs shinobi)
- LEFT vs HIT (shinobi vs shinobi)

*Low-level feature comparisons (6):*
- All pairwise comparisons between: luminance, optical_flow, audio_envelope, button_presses_count

*Low-level vs Shinobi annotation comparisons (8):*
- Kill vs each of the 4 low-level features
- RIGHT vs each of the 4 low-level features

**Total: 18 comparisons when using `--run-all`**

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--cond1` | str | None | No | First condition in format `source:condition` (e.g., `shinobi:Kill` or `hcp:reward`) |
| `--cond2` | str | None | No | Second condition in format `source:condition` (e.g., `shinobi:HealthLoss` or `hcp:punishment`) |
| `--run-all` | flag | False | No | Generate all predefined comparisons (default if no conditions specified) |
| `--threshold` | float | 3.0 | No | Significance threshold for z-maps |
| `--use-raw-maps` | flag | False | No | Use raw (uncorrected) z-maps instead of corrected maps (default: corrected maps) |
| `--exclude-low-level` | flag | False | No | Exclude low-level features from comparisons |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |
| `--output-dir` | str | See below | No | Custom output directory |

**Output Directory Logic:**

**By default, corrected maps are used.** The output directory is automatically determined:
- Default (no flags, corrected maps): `reports/figures_corrected/condition_comparison/`
- `--use-raw-maps` only: `reports/figures_raw/condition_comparison/`

**Common Use Cases:**

```bash
# Generate all predefined comparisons (uses corrected maps by default, outputs to figures_corrected/)
invoke viz.condition-comparison --run-all --verbose 1

# Compare two specific conditions (corrected maps by default)
invoke viz.condition-comparison --cond1 shinobi:Kill --cond2 hcp:reward

# Use custom threshold
invoke viz.condition-comparison --run-all --threshold 2.5

# Exclude low-level features from comparison
invoke viz.condition-comparison --run-all --exclude-low-level

# Compare specific low-level features (corrected by default)
invoke viz.condition-comparison --cond1 shinobi:luminance --cond2 shinobi:optical_flow

# Compare low-level feature with annotation (corrected by default)
invoke viz.condition-comparison --cond1 shinobi:audio_envelope --cond2 shinobi:Kill

# Use RAW (uncorrected) maps instead (outputs to figures_raw/)
invoke viz.condition-comparison --run-all --use-raw-maps

# Exclude low-level features and use raw maps
invoke viz.condition-comparison --run-all --use-raw-maps --exclude-low-level

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
| `--use-corrected-maps` | flag | False | No | Use corrected z-maps instead of raw maps (default: raw maps) |
| `--exclude-low-level` | flag | False | No | Exclude low-level features from tables |
| `--overwrite` | flag | False | No | Overwrite existing cluster files |

**Common Use Cases:**

```bash
# Generate atlas tables with default settings
invoke viz.atlas-tables

# Use custom thresholds
invoke viz.atlas-tables --cluster-extent 10 --voxel-thresh 2.5

# Only positive activations
invoke viz.atlas-tables --direction pos

# Use corrected z-maps instead of raw maps
invoke viz.atlas-tables --use-corrected-maps

# Exclude low-level features from atlas tables
invoke viz.atlas-tables --exclude-low-level

# Use custom input/output directories
invoke viz.atlas-tables --input-dir /path/to/zmaps --output-dir /path/to/tables

# Force overwrite existing files
invoke viz.atlas-tables --overwrite
```

---

### `viz.fingerprinting`

Compute fingerprinting scores (if needed) and generate visualizations.

**Description:**

Fingerprinting assesses whether brain maps are participant-specific by checking if each map's most similar map (nearest neighbor) comes from the same subject. This task automatically computes fingerprinting scores if they don't already exist, then creates comprehensive visualizations.

The computation is fast (runs on the pre-computed correlation matrix) and is integrated into the visualization task for convenience.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**

Computed data (if not already present):
- `processed/fingerprinting/fingerprinting_detailed.tsv` - Per-map fingerprinting results
- `processed/fingerprinting/fingerprinting_aggregated.tsv` - Summary statistics by subject/condition/level

Figures:
- `reports/figures/fingerprinting/fingerprinting_confusion_matrix.png` - Subject identification confusion matrix
- `reports/figures/fingerprinting/fingerprinting_correlation_distributions.png` - Within vs between subject correlations
- `reports/figures/fingerprinting/fingerprinting_by_source.png` - Scores by analysis level
- `reports/figures/fingerprinting/fingerprinting_by_condition.png` - Scores by condition
- `reports/figures/fingerprinting/fingerprinting_nearest_neighbor_correlations.png` - Distribution of nearest neighbor similarities

**Common Use Cases:**

```bash
# Run fingerprinting analysis and generate visualizations (single command)
invoke viz.fingerprinting

# With verbose output to see computation details
invoke viz.fingerprinting --verbose 1

# With custom log directory
invoke viz.fingerprinting --log-dir ./logs/viz_fingerprinting
```

**What it computes and visualizes:**
- Computation: For each map, finds nearest neighbor (highest correlation) and checks if from same subject
- Fingerprinting score = proportion of maps where nearest neighbor is from same subject
- Aggregates by subject, condition, and analysis level
- Visualizations: Confusion matrices, within vs between subject correlation distributions, performance by condition/level

---

### `viz.within-subject-correlations`

Compute and visualize within-subject condition correlations.

**Description:**

Computes how maps from different conditions correlate with each other within each subject, then creates heatmaps and plots demonstrating condition specificity and distinctiveness. This task combines both computation and visualization in a single step (computation is fast, so no need for separate steps).

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--exclude-low-level` | flag | False | No | Exclude low-level features from plots |
| `--verbose` | int | 0 | No | Verbosity level (0-2) |
| `--log-dir` | str | None | No | Custom log directory |

**Outputs:**
- `reports/figures/within_subject_condition_correlations/{subject}_condition_correlations.png` - Per-subject condition correlation heatmaps
- `reports/figures/within_subject_condition_correlations/average_condition_correlations.png` - Average heatmap across subjects
- `reports/figures/within_subject_condition_correlations/same_vs_different_conditions.png` - Same vs different condition comparison
- `reports/figures/within_subject_condition_correlations/condition_specificity_matrix.png` - Specificity scores by condition and subject

**Common Use Cases:**

```bash
# Compute and visualize within-subject condition correlations
invoke viz.within-subject-correlations --verbose 1

# Exclude low-level features from plots
invoke viz.within-subject-correlations --exclude-low-level

# With custom log directory
invoke viz.within-subject-correlations --log-dir ./logs/viz_correlations
```

**What it does:**
- Computes condition × condition correlation matrix for each subject
- Computes average correlation matrices across all subjects
- Generates correlation heatmaps for each subject (separated by Shinobi/HCP)
- Creates violin and box plots comparing same-condition vs different-condition correlations
- Plots condition specificity matrix showing which conditions are most distinctive
- Computes statistical comparisons including Cohen's d and significance tests

---

### `viz.mvpa-confusion-matrices`

Generate publication-quality confusion matrix visualization for all subjects with task icons and separation styling.

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--screening` | int | 20 | No | Screening percentile used |
| `--exclude-low-level` | flag | False | No | Exclude low-level features from confusion matrices |
| `--output` | str | Auto | No | Output path for figure |

**Output:**
- `reports/figures_raw/mvpa_confusion_matrices_s{screening}.png` - 2x2 grid of confusion matrices

**Common Use Cases:**

```bash
# Generate figure with default settings
invoke viz.mvpa-confusion-matrices

# Custom screening percentile
invoke viz.mvpa-confusion-matrices --screening 10

# Exclude low-level features from confusion matrices
invoke viz.mvpa-confusion-matrices --exclude-low-level

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

## Descriptive Statistics Tasks

### `descriptive.viz`

Generate descriptive statistics visualization figure with dataset summary.

Creates publication-ready 3-panel figure:
- **Panel A:** Events by subject and condition (grouped bar chart, spans top row)
- **Panel B:** Session/run availability matrix (heatmap, bottom left)
- **Panel C:** Volume counts distribution (box plot, bottom right)

Automatically generates CSV summary if it doesn't exist.

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path` | str | `{DATA_PATH}` | Path to data directory |
| `--output` | str | `{FIG_PATH}/descriptive_stats.png` | Output figure path |
| `--csv-path` | str | Auto-detect | Path to dataset_summary.csv |
| `--force` | flag | False | Force regeneration of both CSV and figure |
| `--verbose` | int | 0 | Verbosity level (0-2) |
| `--log-dir` | str | None | Custom log directory |

**Common Use Cases:**

```bash
# Generate figure (auto-generates CSV if needed)
invoke descriptive.viz --verbose 1

# Force regeneration of both CSV and figure
invoke descriptive.viz --force --verbose 1

# Use custom output path
invoke descriptive.viz --output reports/figures/custom.png

# Use existing custom CSV file
invoke descriptive.viz --csv-path /path/to/custom_summary.csv
```

**Output:**

- **CSV:** `{DATA_PATH}/processed/descriptive/dataset_summary.csv` - Dataset summary with one row per run containing subject, session, run identifiers, fMRI and events file availability, number of volumes, and event counts for each condition
- **Figure:** `{FIG_PATH}/descriptive_stats.png` - Publication-ready PNG figure (300 DPI) with 3 panels showing events by subject/condition, session availability, and volume distribution

---

## Pipeline Tasks

These tasks run complete analysis pipelines combining multiple steps.

### `pipeline.full`

Run complete analysis pipeline for a single subject/session.

**Pipeline stages:**
1. Session-level GLM
2. Visualizations

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

## Validation Tasks

### `validate.outputs`

Validate shinobi_fmri pipeline outputs for completeness.

**Description:**

Comprehensive validation of all analysis outputs against available input data. This task:
- Scans for all BOLD files in fmriprep and corresponding events.tsv files
- Checks that all expected GLM outputs exist (session-level and subject-level)
- Validates beta correlation matrix completeness
- Checks MVPA results and permutation tests
- Verifies figure and visualization outputs
- Optionally validates file integrity by loading and checking contents

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--subject` | str | None | No | Specific subject to validate (e.g., `sub-01`). Default: all subjects |
| `--analysis-type` | str | `all` | No | Type of analysis to validate (`glm_session`, `glm_subject`, `mvpa`, `correlations`, `figures`, `all`) |
| `--check-integrity` | flag | False | No | If True, validate file contents (load NIfTI, pickle files) - slower but thorough |
| `--output` | str | None | No | Path to save detailed JSON report (optional) |
| `--verbose` | flag | False | No | If True, enable verbose output (INFO level) |
| `--log-dir` | str | None | No | Custom log directory |
| `--low-level-confs` | flag | False | No | Validate outputs from analyses with low-level confounds (checks `processed_low-level/` directory) |

**Common Use Cases:**

```bash
# Validate everything (quick existence check)
invoke validate.outputs

# Validate specific subject
invoke validate.outputs --subject sub-01

# Validate only GLM session-level outputs
invoke validate.outputs --analysis-type glm_session

# Validate only correlation matrix
invoke validate.outputs --analysis-type correlations

# Validate only MVPA results
invoke validate.outputs --analysis-type mvpa

# Validate only figures and visualizations
invoke validate.outputs --analysis-type figures

# Full integrity check (slower but thorough - loads and validates all files)
invoke validate.outputs --check-integrity --verbose

# Generate detailed JSON report
invoke validate.outputs --output validation_report.json --verbose

# Validate specific subject with integrity check
invoke validate.outputs --subject sub-01 --check-integrity --verbose

# Validate outputs from low-level confounds analyses (processed_low-level/)
invoke validate.outputs --low-level-confs --verbose

# Validate low-level GLM outputs with integrity check
invoke validate.outputs --analysis-type glm_session --low-level-confs --check-integrity --verbose
```

**Important Note About Directories:**

The pipeline supports two output directories:
- `processed/` - Standard analyses (8 game conditions: HIT, JUMP, DOWN, LEFT, RIGHT, UP, Kill, HealthLoss)
- `processed_low-level/` - Analyses with low-level confounds (12 conditions: 8 game + 4 low-level features)

**Use `--low-level-confs` flag** if you ran all your analyses with `--low-level-confs` and your outputs are in `processed_low-level/`.

**What is Validated:**

1. **GLM Session-Level:**
   - Scans fmriprep for all available BOLD files
   - Checks corresponding events.tsv files exist
   - Verifies z-maps (raw and corrected) exist for each session/condition
   - Verifies beta maps exist for each session/condition

2. **GLM Subject-Level:**
   - Checks z-maps (raw and corrected) exist for each subject/condition
   - Verifies beta maps exist for each subject/condition

3. **Beta Correlations:**
   - Checks correlation matrix file exists
   - Validates matrix completeness (which pairs have been computed)
   - Reports missing correlation pairs
   - Cross-validates matrix entries against actual beta map files

4. **MVPA:**
   - Checks decoder files exist for each subject
   - Verifies weight maps exist
   - Counts permutation files and reports status
   - Checks aggregated permutation results exist

5. **Figures & Visualizations:**
   - Checks for beta correlation plots
   - Verifies subject-level condition plots exist
   - Checks annotation panels
   - Validates GLM HTML reports
   - Checks MVPA confusion matrices

**Output Example:**

```
================================================================================
VALIDATION SUMMARY
================================================================================

GLM Session-Level:
  Expected: 256
  Found:    250
  Missing:  6
  Complete: 97.7%
  Input Data:
    Subjects: 4
    Runs: 64
    ⚠ Missing events.tsv: 2
  Sessions Checked: 32

GLM Subject-Level:
  Expected: 64
  Found:    64
  Missing:  0
  Complete: 100.0%
  Subjects Checked: 4

MVPA (screening=20):
  Expected: 4
  Found:    4
  Missing:  0
  Complete: 100.0%
  Permutation Files Found:
    sub-01: 100 files
    sub-02: 100 files
    sub-04: 100 files
    sub-06: 100 files

Correlations - Beta Maps:
  Expected: 1
  Found:    1
  Missing:  0
  Complete: 100.0%
  Matrix Details:
    Total maps in matrix: 1024
    Shinobi session-level maps: 256
    Computed pairs: 523776/523776
    Matrix completion: 100.0%

================================================================================
OVERALL:
  Expected: 325
  Found:    319
  Missing:  6
  Complete: 98.2%
================================================================================
```

**Tips:**
- Run this after completing analysis steps to verify outputs
- Use `--check-integrity` for thorough validation before publication
- Save JSON report for documentation and tracking
- Re-run after fixing missing outputs to verify completeness

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
invoke glm.session-level

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
