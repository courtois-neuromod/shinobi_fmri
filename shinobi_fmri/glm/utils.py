"""
GLM Analysis Utilities

This module provides utility functions for General Linear Model (GLM) analysis
of fMRI data, including data loading, preprocessing, design matrix creation,
and statistical inference.

Constants:
    TR: Repetition time in seconds (1.49s for shinobi dataset)
    HRF_MODEL: Hemodynamic response function model ('spm')
"""

import os
import os.path as op
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import nibabel as nib
from nibabel import Nifti1Image
from nilearn import image
from nilearn.image import clean_img
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import cluster_level_inference
from shinobi_fmri import config

# GLM parameters from config
TR = config.GLM_TR
HRF_MODEL = config.GLM_HRF_MODEL


# ============================================================================
# Input Validation Functions
# ============================================================================

def validate_file_exists(file_path: str, file_description: str) -> None:
    """
    Validate that a required file exists with a helpful error message.

    Args:
        file_path: Path to file to check
        file_description: Human-readable description for error message

    Raises:
        FileNotFoundError: If file does not exist, with helpful troubleshooting info

    Example:
        >>> validate_file_exists('/path/to/data.nii.gz', 'fMRI data')
    """
    if not op.exists(file_path):
        error_msg = f"""
{file_description} file not found: {file_path}

Troubleshooting:
  1. Check that the file path is correct
  2. Verify that preprocessing (fMRIPrep) has completed for this subject/session
  3. Check that the data directory is mounted/accessible
  4. Verify BIDS naming conventions are followed

Directory: {op.dirname(file_path)}
Expected filename: {op.basename(file_path)}
"""
        raise FileNotFoundError(error_msg)


def validate_nifti_file(file_path: str, expected_ndim: Optional[int] = None) -> None:
    """
    Validate that a NIfTI file can be loaded and has expected dimensions.

    Args:
        file_path: Path to NIfTI file
        expected_ndim: Expected number of dimensions (3 for anatomical, 4 for fMRI)

    Raises:
        ValueError: If file cannot be loaded or has wrong dimensions

    Example:
        >>> validate_nifti_file('/path/to/fmri.nii.gz', expected_ndim=4)
    """
    try:
        img = nib.load(file_path)
        if expected_ndim is not None and img.ndim != expected_ndim:
            raise ValueError(
                f"NIfTI file has {img.ndim} dimensions, expected {expected_ndim}\n"
                f"File: {file_path}\n"
                f"Shape: {img.shape}"
            )
    except Exception as e:
        raise ValueError(
            f"Failed to load NIfTI file: {file_path}\n"
            f"Error: {str(e)}\n"
            f"Make sure the file is a valid NIfTI (.nii or .nii.gz) file."
        )


def validate_events_file(events_path: str, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate that an events TSV file can be loaded and has required columns.

    Args:
        events_path: Path to events.tsv file
        required_columns: List of required column names (default: ['onset', 'duration'])

    Raises:
        ValueError: If file cannot be loaded or missing required columns

    Example:
        >>> validate_events_file('/path/to/events.tsv', ['onset', 'duration', 'trial_type'])
    """
    if required_columns is None:
        required_columns = ['onset', 'duration']

    try:
        events = pd.read_csv(events_path, sep='\t')
    except Exception as e:
        raise ValueError(
            f"Failed to load events file: {events_path}\n"
            f"Error: {str(e)}\n"
            f"Make sure the file is a valid TSV (tab-separated values) file."
        )

    missing_cols = [col for col in required_columns if col not in events.columns]
    if missing_cols:
        raise ValueError(
            f"Events file missing required columns: {missing_cols}\n"
            f"File: {events_path}\n"
            f"Available columns: {list(events.columns)}\n"
            f"Required columns: {required_columns}"
        )


def validate_contrast_exists(contrast_name: str, design_matrix: pd.DataFrame) -> None:
    """
    Validate that a contrast regressor exists in the design matrix.

    Args:
        contrast_name: Name of contrast/condition to check
        design_matrix: Design matrix DataFrame

    Raises:
        ValueError: If contrast not found in design matrix columns

    Example:
        >>> validate_contrast_exists('HIT', design_matrix)
    """
    if contrast_name not in design_matrix.columns:
        raise ValueError(
            f"Contrast '{contrast_name}' not found in design matrix\n"
            f"Available regressors: {list(design_matrix.columns)}\n"
            f"Check that:\n"
            f"  1. The condition name is spelled correctly\n"
            f"  2. Events for this condition exist in the events.tsv file\n"
            f"  3. The condition is in the conditions_list parameter"
        )


# ============================================================================
# Preprocessing Utilities
# ============================================================================

def get_scrub_regressor(
    run_events: pd.DataFrame,
    design_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Create scrub regressors to exclude timepoints without valid behavioral data.

    Identifies frames outside valid game repetitions (those with available .bk2 files)
    and creates individual scrub confound regressors to mark them for exclusion.
    This ensures only timepoints with behavioral data are included in GLM analysis.

    Args:
        run_events: Events DataFrame containing all events for one run, including
                   'trial_type', 'onset', 'duration', and 'stim_file' columns
        design_matrix: Design matrix created by Nilearn's make_first_level_design_matrix

    Returns:
        Design matrix with added scrub confound columns (scrub1, scrub2, etc.)
        One column per excluded timepoint, with value 1.0 at that timepoint

    Note:
        Repetitions without valid .bk2 files (marked as "Missing file" or NaN in
        stim_file column) are excluded by creating scrub regressors for those frames.

    Example:
        >>> design_matrix_full = get_scrub_regressor(run_events, design_matrix_raw)
        >>> # Frames outside valid reps now have scrub regressors
    """
    reps = []
    # Get repetition segments
    for i in range(len(run_events)):
        if run_events['trial_type'][i] == "gym-retro_game":
            reps.append(run_events.iloc[i, :])

    # Get time vector
    time = np.array(design_matrix.index)

    to_keep = np.zeros(len(time))
    # Generate binary regressor
    for i in range(len(time)):
        for rep in reps:
            if type(rep["stim_file"]) == str and rep["stim_file"] != "Missing file" and type(rep["stim_file"]) != float:
                if time[i] >= rep['onset'] and time[i] <= rep['onset'] + rep['duration']:
                    to_keep[i] = 1.0

    # Collect all scrub regressors first, then concat at once for better performance
    scrub_regressors = {}
    scrub_idx = 1
    for idx, timepoint in enumerate(to_keep):
        if timepoint == 0.0:  # If to_keep is zero create a scrub regressor to remove this frame
            scrub_regressor = np.zeros(len(time))
            scrub_regressor[idx] = 1.0
            scrub_regressors[f'scrub{scrub_idx}'] = scrub_regressor
            scrub_idx += 1

    # Add all scrub regressors at once using pd.concat instead of repeated insertion
    if scrub_regressors:
        scrub_df = pd.DataFrame(scrub_regressors, index=design_matrix.index)
        design_matrix = pd.concat([design_matrix, scrub_df], axis=1)

    return design_matrix


# ============================================================================
# File Path Construction
# ============================================================================

def get_filenames(
    sub: str,
    ses: str,
    run: str,
    path_to_data: str
) -> Tuple[str, str, str, str]:
    """
    Construct BIDS-compliant file paths for fMRI data and associated files.

    Args:
        sub: Subject identifier (e.g., 'sub-01')
        ses: Session identifier (e.g., 'ses-001')
        run: Run number (e.g., '1' or '01')
        path_to_data: Root path to data directory

    Returns:
        Tuple of (fmri_fname, anat_fname, events_fname, mask_fname):
            - fmri_fname: Path to preprocessed fMRI NIfTI file
            - anat_fname: Path to preprocessed T1w anatomical image
            - events_fname: Path to annotated events TSV file
            - mask_fname: Path to brain mask NIfTI file

    Note:
        Files are expected to follow BIDS naming conventions and reside in
        the fMRIPrep output structure.
    """
    # Ensure run is zero-padded to 2 digits for event files
    run_padded = f"{int(run):02d}"

    fmri_fname = op.join(
        path_to_data,
        "shinobi.fmriprep",
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    # assert op.isfile(fmri_fname), f"fMRI file not found for {sub}_{ses}_{run}"
    
    anat_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
    )
    # assert op.isfile(anat_fname), f"sMRI file not found for {sub}_{ses}_{run}"
    
    mask_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    # assert op.isfile(mask_fname), f"Mask file not found for {sub}_{ses}_{run}"
    
    events_fname = op.join(
        path_to_data,
        "shinobi",
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-{run_padded}_desc-annotated_events.tsv",
    )
    # assert op.isfile(events_fname), f"Annotated events file not found for {sub}_{ses}_{run}"
    
    return fmri_fname, anat_fname, events_fname, mask_fname

def load_image_and_mask(
    fmri_fname: str,
    mask_fname: str,
    t_r: float = TR
) -> Tuple[Nifti1Image, Nifti1Image]:
    """
    Load fMRI data and brain mask, applying preprocessing and resampling.

    Loads a 4D fMRI NIfTI image, applies standardization and detrending,
    and resamples the anatomical brain mask to match the functional image
    geometry.

    Args:
        fmri_fname: Path to fMRI NIfTI file (.nii.gz)
        mask_fname: Path to brain mask NIfTI file (.nii.gz)
        t_r: Repetition time in seconds (default: 1.49)

    Returns:
        Tuple of (fmri_img, mask_resampled):
            - fmri_img: Preprocessed 4D fMRI image (standardized, detrended)
            - mask_resampled: Brain mask resampled to functional image space

    Note:
        The fMRI image is standardized (z-scored) and linearly detrended.
        No high-pass filtering is applied (handled via design matrix).
        The mask is resampled to 4mm isotropic voxels using nearest-neighbor
        interpolation.
    """
    # Load and clean 4d image
    fmri_img = clean_img(
        fmri_fname,
        standardize=True,
        detrend=True,
        high_pass=None,
        t_r=t_r,
        ensure_finite=True,
        confounds=None,
    )
    # Load and resample (i.e. morph  ?) anat mask
    aff_orig = nib.load(fmri_fname).affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    
    # Handle different calling conventions or image types
    # compute_run_level uses fmri_img.get_fdata().shape[:3]
    # compute_session_level uses fmri_img.shape[:3]
    # clean_img returns a Nifti1Image, so .shape is correct.
    target_shape = fmri_img.shape[:3]
    
    mask_resampled = image.resample_img(
        mask_fname,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation='nearest',
        force_resample=True,
        copy_header=True,
    )
    return fmri_img, mask_resampled

def downsample_to_TR(
    signal: np.ndarray,
    fs: float = 60.0,
    TR: float = TR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample a high-frequency signal to match fMRI repetition time.

    Downsamples behavioral or psychophysical signals (typically sampled at 60 Hz)
    to match fMRI acquisition timing by averaging within TR bins.

    Args:
        signal: 1D array of signal values sampled at high frequency
        fs: Original sampling rate in Hz (default: 60.0)
        TR: Target repetition time in seconds (default: 1.49)

    Returns:
        Tuple of (downsampled_signal, timepoints):
            - downsampled_signal: Signal averaged within TR bins
            - timepoints: Time of each bin center in seconds

    Example:
        >>> signal = np.random.randn(600)  # 10 seconds at 60 Hz
        >>> ds_signal, times = downsample_to_TR(signal, fs=60.0, TR=1.49)
        >>> len(ds_signal)  # Should be ~7 TRs
        7
    """
    n = len(signal)
    t = np.arange(n) / fs  # time of each original sample
    bin_idx = (t // TR).astype(int)  # which TR-bin each sample falls into

    # Average within bins using bincount
    sums = np.bincount(bin_idx, weights=signal)
    counts = np.bincount(bin_idx)
    ds = sums / counts  # bin means

    # Time of each downsampled point (bin centers)
    t_ds = (np.arange(len(ds)) + 0.5) * TR

    return ds, t_ds

def add_button_press_confounds(
    confounds: Tuple[pd.DataFrame, List[str]],
    run_events: pd.DataFrame,
    t_r: float = TR
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add game controller button press regressors to confound matrix.

    Detects button press and release events from frame-by-frame game state data
    and creates regressors indicating the number of button presses per TR.
    Useful for modeling motor-related variance in video game fMRI.

    Args:
        confounds: Tuple of (confounds_df, column_names) from fMRIPrep
        run_events: DataFrame with trial_type='frame' rows containing button states
        t_r: Repetition time in seconds (default: 1.49)

    Returns:
        Updated confounds tuple with button press regressors added

    Note:
        Adds 'button_presses_count' regressor tracking number of simultaneous
        button presses per TR. Handles 8 buttons: DOWN, LEFT, RIGHT, UP, C, Y, X, Z.
        Button state transitions (False→True) are detected as presses.
    """
    n_volumes = len(confounds[0])
    
    # Button columns to extract
    button_columns = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'C', 'Y', 'X', 'Z']
    
    # Get only frame events - reset index to ensure we can iterate properly
    frame_events = run_events[run_events['trial_type'] == 'frame'].copy().reset_index(drop=True)
    
    if len(frame_events) == 0:
        return confounds

    # Initialize regressors
    any_press = np.zeros(n_volumes)  # 1 if any button pressed
    any_release = np.zeros(n_volumes)  # 1 if any button released
    count_press = np.zeros(n_volumes)  # number of buttons pressed
    count_release = np.zeros(n_volumes)  # number of buttons released
    
    # For each button, detect transitions (press/release events)
    for button in button_columns:
        if button not in frame_events.columns:
            continue
        
        # Get button states as boolean array
        button_states = frame_events[button].copy()
        
        # Convert to boolean, handling NaN and various types
        button_states = button_states.fillna(False)
        if button_states.dtype == 'object':
            button_states = button_states.replace({'False': False, 'True': True, False: False, True: True})
        button_states = button_states.astype(bool)
        
        button_states = button_states.values
        onsets = frame_events['onset'].values

        # Filter out None/NaN onsets
        valid_mask = pd.notna(onsets)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logging.warning(f"Found {n_invalid} frame events with NaN onsets (button {button})")

        onsets = onsets[valid_mask]
        button_states = button_states[valid_mask]

        # Detect transitions: False->True = press, True->False = release
        presses = np.zeros(len(button_states), dtype=bool)
        releases = np.zeros(len(button_states), dtype=bool)

        for i in range(1, len(button_states)):
            if not button_states[i-1] and button_states[i]:
                presses[i] = True
            elif button_states[i-1] and not button_states[i]:
                releases[i] = True
        
        # Map events to volumes (TR bins)
        for i, onset in enumerate(onsets):
            volume_idx = int(np.round(onset / t_r))
            if 0 <= volume_idx < n_volumes:
                if presses[i]:
                    any_press[volume_idx] = 1
                    count_press[volume_idx] += 1
                if releases[i]:
                    any_release[volume_idx] = 1
                    count_release[volume_idx] += 1
    
    # Add regressors to confounds[0]
    #confounds[0]['button_any_press'] = any_press
    #confounds[0]['button_any_release'] = any_release
    confounds[0]['button_presses_count'] = count_press
    #confounds[0]['button_count_release'] = count_release
    
    return confounds

def add_psychophysics_confounds(
    confounds: Tuple[pd.DataFrame, List[str]],
    run_events: pd.DataFrame,
    path_to_data: str,
    t_r: float = TR
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add low-level visual and audio features as confound regressors.

    Loads precomputed psychophysical features (e.g., luminance, contrast,
    audio envelope) from game recordings and downsamples them to TR timing
    for use as nuisance regressors.

    Args:
        confounds: Tuple of (confounds_df, column_names) from fMRIPrep
        run_events: DataFrame containing trial_type='gym-retro_game' events
        path_to_data: Root path to data directory
        t_r: Repetition time in seconds (default: 1.49)

    Returns:
        Updated confounds tuple with psychophysical regressors added

    Note:
        Expects .npy files with precomputed features at 60 Hz sampling rate.
        Features are downsampled to TR using bin averaging.
        Missing files are skipped with a warning.
    """
    n_volumes = len(confounds[0])
    ppc_data = {}  # Dictionary to store accumulated data for each key
    n_invalid_onsets = 0

    for idx, row in run_events.iterrows():
        if row["trial_type"] != "gym-retro_game":
            continue

        if pd.isna(row["onset"]) or pd.isna(row["stim_file"]):
            n_invalid_onsets += 1
            continue

        ppc_fname = op.join(
            path_to_data, "shinobi", row["stim_file"].replace(".bk2", "_confs.npy")
        )

        if not os.path.exists(ppc_fname):
            continue

        try:
            ppc_rep = np.load(ppc_fname, allow_pickle=True).item()
        except Exception as e:
            logging.warning(f"Failed to load low-level features from {ppc_fname}: {e}")
            continue

        onset = float(row["onset"])
        onset_offset = int(round(onset / t_r))

        for key, val in ppc_rep.items():
            x = np.asarray(val, dtype=float)

            # Downsample to TR
            y_tr, t_ds = downsample_to_TR(x, fs=60.0, TR=t_r)

            if key not in ppc_data:
                ppc_data[key] = np.zeros(n_volumes)

            end_idx = min(onset_offset + len(y_tr), n_volumes)
            valid_length = end_idx - onset_offset
            if valid_length > 0:
                ppc_data[key][onset_offset:end_idx] += y_tr[:valid_length]

    for key, data in ppc_data.items():
        confounds[0][key] = data

    if n_invalid_onsets > 0:
        logging.warning(f"Found {n_invalid_onsets} gym-retro_game events with NaN onsets (low-level features)")

    return confounds

def get_clean_matrix(
    fmri_fname: str,
    fmri_img: Nifti1Image,
    annotation_events: pd.DataFrame,
    run_events: pd.DataFrame,
    path_to_data: str,
    use_low_level_confs: bool = False,
    t_r: float = TR,
    hrf_model: str = HRF_MODEL
) -> pd.DataFrame:
    """
    Create a complete GLM design matrix with confounds and task regressors.

    Loads fMRIPrep confounds (motion, white matter, CSF, global signal),
    optionally adds low-level psychophysical and button press confounds,
    convolves task events with HRF, and creates scrubbing regressors for
    high-motion volumes.

    Args:
        fmri_fname: Path to fMRI file (for loading fMRIPrep confounds)
        fmri_img: Preprocessed fMRI image (for getting dimensions)
        annotation_events: DataFrame with trial_type, onset, duration columns
        run_events: Full events DataFrame with all trial types
        path_to_data: Root path to data directory
        use_low_level_confs: Whether to include psychophysical confounds (default: False)
        t_r: Repetition time in seconds (default: 1.49)
        hrf_model: HRF model name (default: 'spm')

    Returns:
        Complete design matrix with all regressors (task + confounds + scrubbing)

    Note:
        Confounds include: motion (24 parameters), white matter, CSF,
        global signal, and optionally low-level features and button presses.
        Scrubbing regressors are added for volumes exceeding motion thresholds.
    """
    import nilearn.interfaces.fmriprep
    # Load confounds
    confounds = nilearn.interfaces.fmriprep.load_confounds(
        fmri_fname,
        strategy=("motion", "high_pass", "wm_csf"),
        motion="full",
        wm_csf="basic",
        global_signal="full",
    )

    if use_low_level_confs:
        confounds = add_psychophysics_confounds(confounds, run_events, path_to_data, t_r=t_r)
        confounds = add_button_press_confounds(confounds, run_events, t_r=t_r)

    # Generate design matrix
    bold_shape = fmri_img.shape
    n_slices = bold_shape[-1]
    frame_times = np.arange(n_slices) * t_r
    design_matrix_raw = make_first_level_design_matrix(
        frame_times,
        events=annotation_events,
        drift_model=None,
        hrf_model=hrf_model,
        add_regs=confounds[0],
        add_reg_names=confounds[0].keys(),
    )

    design_matrix_full = get_scrub_regressor(run_events, design_matrix_raw)
    return design_matrix_full

def make_and_fit_glm(
    fmri_imgs: List[Nifti1Image],
    design_matrices: List[pd.DataFrame],
    mask_resampled: Nifti1Image,
    t_r: float = TR,
    hrf_model: str = HRF_MODEL
) -> FirstLevelModel:
    """
    Fit a first-level GLM to fMRI data using nilearn.

    Creates and fits a GLM with AR(1) noise model, spatial smoothing,
    and the specified HRF model.

    Args:
        fmri_imgs: List of preprocessed fMRI images (one per run)
        design_matrices: List of design matrices (one per run)
        mask_resampled: Brain mask in functional space
        t_r: Repetition time in seconds (default: 1.49)
        hrf_model: HRF model name (default: 'spm')

    Returns:
        Fitted FirstLevelModel object

    Note:
        Model configuration:
        - Noise model: AR(1) autocorrelation
        - Smoothing FWHM: 5mm isotropic
        - No drift model (handled via confounds)
        - 16 parallel jobs for efficiency
        - Standardize: False (data pre-standardized)
    """
    fmri_glm = FirstLevelModel(
        t_r=t_r,
        noise_model=config.GLM_NOISE_MODEL,
        standardize=False,
        hrf_model=hrf_model,
        drift_model=config.GLM_DRIFT_MODEL,
        high_pass=None,
        n_jobs=config.GLM_N_JOBS,
        smoothing_fwhm=config.GLM_SMOOTHING_FWHM,
        mask_img=mask_resampled,
        minimize_memory=False,
    )
    fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)
    return fmri_glm

def make_z_map(
    z_map_fname: str,
    beta_map_fname: str,
    report_fname: str,
    fmri_glm: FirstLevelModel,
    regressor_name: str,
    cluster_thresh: Optional[float] = 2.3,
    alpha: float = 0.05
) -> Nifti1Image:
    """
    Compute and save statistical maps for a GLM contrast.

    Computes beta (effect size), z-score, and cluster-corrected z-maps
    for a specified regressor. Also generates an HTML report with
    statistical results and visualizations.

    Args:
        z_map_fname: Output path for uncorrected z-map (.nii.gz)
        beta_map_fname: Output path for beta map (.nii.gz)
        report_fname: Output path for HTML report
        fmri_glm: Fitted FirstLevelModel object
        regressor_name: Name of regressor/contrast to compute
        cluster_thresh: Z-score threshold for cluster detection (default: 2.3)
                       Set to None to skip cluster correction
        alpha: Family-wise error rate for cluster correction (default: 0.05)

    Returns:
        Uncorrected z-score map as Nifti1Image

    Raises:
        ValueError: If regressor_name not found in design matrix

    Note:
        Statistical inference:
        - Beta maps: Raw parameter estimates (effect sizes)
        - Z-maps: F-test z-scores (uncorrected)
        - Corrected maps: Cluster-level FWE correction at specified alpha
        - Cluster correction uses height threshold + extent test

        Output files:
        - {base}_stat-z.nii.gz: Uncorrected z-map
        - {base}_stat-beta.nii.gz: Beta (effect size) map
        - {base}_desc-corrected_stat-z.nii.gz: Cluster-corrected z-map
        - {base}_report.html: Interactive visualization
    """
    # Check if regressor exists in the design matrix
    design_matrix = fmri_glm.design_matrices_[0]
    if regressor_name not in design_matrix.columns:
        raise ValueError(f"Regressor '{regressor_name}' not found in design matrix (condition did not occur in this run/session)")

    # Get betas
    beta_map = fmri_glm.compute_contrast(regressor_name, output_type="effect_size")
    beta_map.to_filename(beta_map_fname)

    # Get Z_map
    z_map = fmri_glm.compute_contrast(
        regressor_name, output_type="z_score", stat_type="F"
    )
    z_map.to_filename(z_map_fname)

    # Compute and save cluster-corrected Z-map
    # NOTE: cluster_level_inference returns a p-value map, we need to threshold it
    # to get a proper z-map with original z-values in significant clusters
    if cluster_thresh is not None:
        try:
            # Get cluster-level FWE-corrected p-value map
            p_map = cluster_level_inference(z_map, threshold=cluster_thresh, alpha=alpha)
            p_data = p_map.get_fdata()
            z_data = z_map.get_fdata()

            # Create thresholded z-map: keep original z-values only for voxels in significant clusters
            sig_mask = (p_data > 0) & (p_data < alpha)
            thresholded_z = np.zeros_like(z_data)
            thresholded_z[sig_mask] = z_data[sig_mask]

            # Save properly thresholded z-map
            corrected_fname = z_map_fname.replace('_stat-z.nii.gz', '_desc-corrected_stat-z.nii.gz')
            corrected_img = Nifti1Image(thresholded_z, z_map.affine, z_map.header)
            corrected_img.to_filename(corrected_fname)

            # Log success (basic, since this function doesn't have logger)
            n_sig_voxels = np.sum(sig_mask)
            if n_sig_voxels > 0:
                print(f"  FWE correction: {n_sig_voxels} voxels survive")
            else:
                print(f"  FWE correction: No significant clusters")

        except Exception as e:
            print(f"Warning: Failed to compute cluster correction for {regressor_name}: {e}")

    # Get report
    report = fmri_glm.generate_report(
        contrasts=[regressor_name],
        height_control=None
    )
    report.save_as_html(report_fname)
    return z_map

def select_events(
    run_events: pd.DataFrame,
    conditions_list: List[str]
) -> pd.DataFrame:
    """
    Filter events DataFrame to include only specified conditions.

    Extracts events corresponding to task conditions of interest,
    returning only the columns needed for GLM design matrix creation.

    Args:
        run_events: Complete events DataFrame with all trial types
        conditions_list: List of condition names to include (e.g., ['HIT', 'JUMP'])

    Returns:
        Filtered DataFrame with columns: trial_type, onset, duration

    Example:
        >>> events = pd.DataFrame({
        ...     'trial_type': ['HIT', 'frame', 'JUMP', 'HIT'],
        ...     'onset': [1.0, 1.5, 3.0, 5.0],
        ...     'duration': [0.5, 0.0, 0.5, 0.5]
        ... })
        >>> select_events(events, ['HIT', 'JUMP'])
        # Returns only HIT and JUMP rows with onset/duration
    """
    annotation_events = run_events[run_events["trial_type"].isin(conditions_list)]
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]
    return annotation_events

def load_run(
    fmri_fname: str,
    mask_fname: str,
    events_fname: str,
    path_to_data: str,
    conditions_list: List[str],
    use_low_level_confs: bool = False
) -> Tuple[pd.DataFrame, Nifti1Image, Nifti1Image]:
    """
    Load and prepare all data needed for run-level GLM analysis.

    Orchestrates loading of fMRI data, events, masks, and creation of the
    complete design matrix with confounds. This is the main data preparation
    function called before fitting a GLM.

    Args:
        fmri_fname: Path to preprocessed fMRI file (.nii.gz)
        mask_fname: Path to brain mask file (.nii.gz)
        events_fname: Path to events TSV file
        path_to_data: Root path to data directory
        conditions_list: List of condition names to model
        use_low_level_confs: Whether to include psychophysical confounds (default: False)

    Returns:
        Tuple of (design_matrix, fmri_img, mask_resampled):
            - design_matrix: Complete design matrix with task + confounds
            - fmri_img: Preprocessed 4D fMRI image
            - mask_resampled: Brain mask in functional space

    Example:
        >>> conditions = ['HIT', 'JUMP', 'Kill']
        >>> dm, img, mask = load_run(
        ...     '/path/to/fmri.nii.gz',
        ...     '/path/to/mask.nii.gz',
        ...     '/path/to/events.tsv',
        ...     '/path/to/data',
        ...     conditions
        ... )
        >>> dm.shape  # (n_volumes, n_regressors)
        (240, 45)
    """
    # Validate input files exist
    validate_file_exists(fmri_fname, "fMRI data")
    validate_file_exists(mask_fname, "Brain mask")
    validate_file_exists(events_fname, "Events")

    # Validate file formats
    validate_nifti_file(fmri_fname, expected_ndim=4)
    validate_nifti_file(mask_fname, expected_ndim=3)
    validate_events_file(events_fname, required_columns=['onset', 'duration'])

    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0], low_memory=False)
    annotation_events = select_events(run_events, conditions_list)

    # Load images
    fmri_img, mask_resampled = load_image_and_mask(fmri_fname, mask_fname)

    # Make design matrix
    design_matrix_clean = get_clean_matrix(
        fmri_fname, fmri_img, annotation_events, run_events, path_to_data, use_low_level_confs=use_low_level_confs
    )
    return design_matrix_clean, fmri_img, mask_resampled
