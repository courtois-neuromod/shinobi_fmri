# Changelog

All notable changes to the shinobi_fmri project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Low-level sensory/motor feature analysis**: Complete pipeline support for analyzing brain responses to low-level visual, audio, and motor features as an alternative to game conditions
  - **New condition set**: `luminance`, `optical_flow`, `audio_envelope`, `button_presses_count`
  - **Configuration**: Added `low_level_conditions` list to `config.yaml` and `config.yaml.template`
  - **GLM analysis**: Modified session-level and subject-level GLM scripts to model low-level features as task regressors (with HRF convolution) when `--low-level-confs` flag is used
  - **HRF convolution**: Created `add_low_level_task_regressors()` function in `glm/utils.py` to properly convolve continuous sensory/motor signals with hemodynamic response function
  - **Output separation**: Low-level analyses automatically route to `processed_low-level/` directory to avoid overwriting game condition results
  - **MVPA support**: Updated `compute_mvpa.py` to support `--low-level-confs` flag for pattern classification on sensory features
  - **Correlation analysis**: Updated `compute_beta_correlations.py` to process low-level feature beta maps
  - **Visualization support**: All visualization scripts (annotation panels, condition comparisons, etc.) now work with low-level conditions via `--low-level-confs` flag
  - **Documentation**: Added "Condition Sets" section to README explaining dual condition system and usage

### Changed
- **Visualization naming**: Standardized all visualization scripts to follow `viz_*.py` naming pattern with underscores instead of hyphens. This improves code consistency and maintainability across the visualization module.
  - Renamed: `beta_correlations_plot.py` → `viz_beta_correlations.py`
  - Renamed: `fingerprinting_plot.py` → `viz_fingerprinting.py`
  - Renamed: `generate_atlas_tables.py` → `viz_atlas_tables.py`
  - Renamed: `mvpa_confusion_matrices.py` → `viz_mvpa_confusion_matrices.py`
  - Renamed: `within_subject_correlations.py` → `viz_within_subject_correlations.py`
  - Renamed: `viz_session-level.py` → `viz_session_level.py`
  - Renamed: `viz_subject-level.py` → `viz_subject_level.py`

### Removed
- **Run-level GLM analysis**: Removed run-level (first-level) GLM computation and visualization. Session-level GLM computes directly from preprocessed fMRI data rather than building on run-level outputs, making run-level analysis redundant.
  - Removed `compute_run_level.py` script and associated SLURM submission scripts
  - Removed `viz_run-level.py` visualization script
  - Removed `glm.run-level` and `viz.run-level` tasks from invoke interface
  - Updated pipeline documentation to reflect simplified workflow
  - Updated validation tests to remove run-level output checks

### Fixed
- **Dataset description metadata**: Corrected session-level GLM dataset description to accurately reflect that it derives from "Preprocessed fMRI data (shinobi.fmriprep)" rather than incorrectly stating "Run-level GLM z-maps"
- **Provenance tracking**: Fixed git commit hash always being null in metadata files. The repo root detection now falls back to searching from the current working directory when `.git` is not found relative to the script path. This ensures git information is captured correctly when running via SLURM or when using virtual environments located outside the repository.
- **Low-level feature NaN/Inf handling**: Fixed GLM failure when using `--low-level-confs` flag due to NaN and Inf values in psychophysical feature regressors. Added comprehensive data validation and cleaning at three stages: (1) source data loading from `.npy` files, (2) after downsampling to TR (preventing division by zero in empty bins), and (3) after HRF convolution. Invalid values are replaced with 0 and logged as warnings. This ensures low-level feature GLM analyses (luminance, optical_flow, audio_envelope, button_presses_count) complete successfully without "array must not contain infs or NaNs" errors.

