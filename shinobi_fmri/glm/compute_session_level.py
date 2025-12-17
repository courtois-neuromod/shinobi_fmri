import os
import os.path as op
import pandas as pd
from nilearn import image
from shinobi_fmri.annotations.annotations import get_scrub_regressor
from shinobi_fmri.utils.logger import ShinobiLogger
import numpy as np
import argparse
import shinobi_behav
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import clean_img
from nilearn.signal import clean
import nibabel as nib
import logging
import pickle
import nilearn
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress informational warnings
warnings.filterwarnings('ignore', message='.*imgs are being resampled to the mask_img resolution.*')
warnings.filterwarnings('ignore', message='.*Mean values of 0 observed.*')
warnings.filterwarnings('ignore', message='.*design matrices are supplied.*')
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-02",
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-ses",
    "--session",
    default="ses-006",
    type=str,
    help="Session to process",
)
parser.add_argument(
    "-v", "--verbose",
    action="count",
    default=0,
    help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
)
parser.add_argument(
    "--log-dir",
    default=None,
    help="Directory for log files",
)
parser.add_argument(
    "--save-glm",
    action="store_true",
    help="Save GLM objects to disk (default: False, only save z-maps)",
)
parser.add_argument(
    "--low-level-confs",
    action="store_true",
    help="Include low-level confounds and button-press rate in design matrix (default: False)",
)
args = parser.parse_args()


def get_filenames(
    sub: str, ses: str, run: str, path_to_data: str
) -> Tuple[str, str, str, str]:
    """
    Returns file names for fMRI, anatomy and annotation events.
    Parameters
    ----------
    subject : str
        Subject id
    session : str
        Session id
    run : str
        Run number (as extracted from fMRI filename, e.g., "1" or "01")
    path_to_data : str
        Path to the data folder
    Returns
    -------
    tuple
        Tuple containing file names of fMRI, anatomy, and annotated events
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
    assert op.isfile(fmri_fname), f"fMRI file not found for {sub}_{ses}_{run}"
    anat_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
    )
    assert op.isfile(anat_fname), f"sMRI file not found for {sub}_{ses}_{run}"
    mask_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    assert op.isfile(mask_fname), f"Mask file not found for {sub}_{ses}_{run}"
    events_fname = op.join(
        path_to_data,
        "shinobi",
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-{run_padded}_desc-annotated_events.tsv",
    )
    assert op.isfile(
        events_fname
    ), f"Annotated events file not found for {sub}_{ses}_{run}"
    return fmri_fname, anat_fname, events_fname, mask_fname


def get_output_names(sub, ses, regressor_output_name, n_runs=None, use_low_level_confs=False):
    """
    Constructs and returns the file paths for the GLM, z-map, beta-map, and report files in BIDS format.

    This function constructs the file paths for the GLM (general linear model),
    z-map, beta-map, and report files for a given subject, session, and regressor, using BIDS-compliant
    directory structure and file naming convention.

    Parameters:
    sub : str
        The subject identifier.
    ses : str
        The session identifier.
    regressor_output_name : str
        The name of the regressor, used in the naming of the output files.
    n_runs : int, optional
        Number of runs to include. If None, uses all runs (session-level, no desc).
        If specified, adds desc-Nruns to filename (e.g., desc-3runs).
    use_low_level_confs : bool
        Whether to include low-level confounds (default: False).
        If True, uses processed_low-level directory (directory name indicates low-level confounds).

    Returns:
    glm_fname : str
        The file path for the GLM file.
    z_map_fname : str
        The file path for the z-map file.
    beta_map_fname : str
        The file path for the beta-map file.
    report_fname : str
        The file path for the report file.
    """
    # Determine output directory based on whether low-level confounds are used
    output_dir = "processed_low-level" if use_low_level_confs else "processed"

    # BIDS-compliant directory structure
    func_dir = op.join(shinobi_behav.DATA_PATH, output_dir, sub, ses, "func")
    os.makedirs(func_dir, exist_ok=True)

    # Optional descriptor for incremental analysis (number of runs)
    # Note: low-level confounds info is already in the directory name (processed vs processed_low-level)
    desc_suffix = f"_desc-{n_runs}runs" if n_runs is not None else ""

    # BIDS-compliant base filename
    base_name = f"{sub}_{ses}_task-shinobi{desc_suffix}_contrast-{regressor_output_name}"

    glm_fname = op.join(func_dir, f"{base_name}_glm.pkl")
    z_map_fname = op.join(func_dir, f"{base_name}_stat-z.nii.gz")
    beta_map_fname = op.join(func_dir, f"{base_name}_stat-beta.nii.gz")

    # Report filename (keeping reports in figures_path)
    if n_runs is None:
        level_dir = "ses-level"
    else:
        level_dir = f"ses-level_{n_runs}run"

    report_dir = op.join(shinobi_behav.FIG_PATH, level_dir, regressor_output_name, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_fname = op.join(report_dir, f"{base_name}_report.html")

    return glm_fname, z_map_fname, beta_map_fname, report_fname


def load_image_and_mask(fmri_fname, mask_fname):
    """
    Load and clean 4d image and resample anat mask
    Parameters
    ----------
    fmri_fname : str
        File name of fMRI image
    anat_fname : str
        File name of anatomy image
    Returns
    -------
    tuple
        Tuple containing cleaned fMRI and resampled anatomy images
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

def add_button_press_confounds(confounds, run_events):
    """
    Add button press/release event regressors to the confounds dataframe.
    Creates regressors for:
    - binary indicator if any button was pressed in a volume
    - binary indicator if any button was released in a volume
    - count of buttons pressed in a volume
    - count of buttons released in a volume
    
    Parameters
    ----------
    confounds : tuple
        Tuple containing confounds dataframe
    run_events : dataframe
        Dataframe containing run events with button press columns
        
    Returns
    -------
    confounds : tuple
        Updated confounds with button press regressors added
    """
    n_volumes = len(confounds[0])
    
    # Button columns to extract
    button_columns = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'C', 'Y', 'X', 'Z']
    
    # Get only frame events - reset index to ensure we can iterate properly
    frame_events = run_events[run_events['trial_type'] == 'frame'].copy().reset_index(drop=True)
    
    if len(frame_events) == 0:
        # print("Warning: No frame events found in run_events")
        return confounds

    # print(f"Found {len(frame_events)} frame events")
    
    # Initialize regressors
    any_press = np.zeros(n_volumes)  # 1 if any button pressed
    any_release = np.zeros(n_volumes)  # 1 if any button released
    count_press = np.zeros(n_volumes)  # number of buttons pressed
    count_release = np.zeros(n_volumes)  # number of buttons released
    
    # For each button, detect transitions (press/release events)
    for button in button_columns:
        
        # Get button states as boolean array
        button_states = frame_events[button].copy()
        
        # Convert to boolean, handling NaN and various types
        button_states = button_states.fillna(False)
        # Handle string 'False'/'True' if present
        if button_states.dtype == 'object':
            button_states = button_states.replace({'False': False, 'True': True, False: False, True: True})
        button_states = button_states.astype(bool)
        
        button_states = button_states.values
        onsets = frame_events['onset'].values

        # Filter out None/NaN onsets and log if any found
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
    confounds[0]['button_count_press'] = count_press
    #confounds[0]['button_count_release'] = count_release
    
    return confounds

def downsample_to_TR(signal, fs=60.0, TR=1.49):
    """
    signal : 1D np.array sampled at 60 Hz
    fs     : original sampling rate (Hz)
    TR     : target sampling period (seconds), e.g. 1.49

    Returns
    -------
    ds     : downsampled signal (bin averages)
    t_ds   : time points of each bin center (seconds)
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

def add_psychophysics_confounds(confounds, run_events):
    """
    Add low-level features confounds to the confounds dataframe
    """
    n_volumes = len(confounds[0])
    ppc_data = {}  # Dictionary to store accumulated data for each key
    n_invalid_onsets = 0

    for idx, row in run_events.iterrows():
        if row["trial_type"] != "gym-retro_game":
            continue

        # Skip if onset is None or NaN and count occurrences
        if pd.isna(row["onset"]):
            n_invalid_onsets += 1
            continue

        ppc_fname = op.join(
            path_to_data, "shinobi", row["stim_file"].replace(".bk2", "_confs.npy")
        )

        if not os.path.exists(ppc_fname):
            continue

        # Try to load low-level features, skip if corrupted
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

            # Initialize array for this key if not exists
            if key not in ppc_data:
                ppc_data[key] = np.zeros(n_volumes)

            # Position the resampled time series in the correct location
            end_idx = min(onset_offset + len(y_tr), n_volumes)
            valid_length = end_idx - onset_offset
            if valid_length > 0:
                ppc_data[key][onset_offset:end_idx] += y_tr[:valid_length]

    # Add ppc columns to confounds[0]
    for key, data in ppc_data.items():
        confounds[0][key] = data

    # Log if any invalid onsets were found
    if n_invalid_onsets > 0:
        logging.warning(f"Found {n_invalid_onsets} gym-retro_game events with NaN onsets (low-level features)")

    return confounds

def get_clean_matrix(fmri_fname, fmri_img, annotation_events, run_events, use_low_level_confs=False):
    """
    Load confounds, create design matrix and return a cleaned matrix
    Parameters
    ----------
    fmri_fname : str
        File name of fMRI image
    fmri_img : nifti-image
        Cleaned fMRI image
    annotation_events : dataframe
        Dataframe containing annotation events
    run_events : dataframe
        Dataframe containing run events
    use_low_level_confs : bool
        Whether to include low-level confounds and button-press rate (default: False)
    Returns
    -------
    matrix
        Design matrix after cleaning
    """
    # Load confounds
    confounds = nilearn.interfaces.fmriprep.load_confounds(
        fmri_fname,
        strategy=("motion", "high_pass", "wm_csf"),
        motion="full",
        wm_csf="basic",
        global_signal="full",
    )
    if use_low_level_confs:
        confounds = add_psychophysics_confounds(confounds, run_events)
        confounds = add_button_press_confounds(confounds, run_events)
    
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

    design_matrix_clean = get_scrub_regressor(run_events, design_matrix_raw)
    # design_matrix_clean = design_matrix_clean.drop(labels="constant", axis=1) ## REMOVE ?
    return design_matrix_clean



def make_and_fit_glm(fmri_imgs, design_matrices, mask_resampled):
    """
    Perform GLM analysis and threshold the results
    Parameters
    ----------
    fmri_img : nifti-image
        Cleaned fMRI image
    mask_resampled : nifti-image
        Resampled mask
    cleaned_matrix : pandas.DataFrame
        Design matrix after cleaning
    Returns
    -------
    None
    """
    fmri_glm = FirstLevelModel(
        t_r=t_r,
        noise_model="ar1",
        standardize=False,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=None,
        n_jobs=16,
        smoothing_fwhm=5,
        mask_img=mask_resampled,
        minimize_memory=False,
    )
    fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)
    return fmri_glm


def make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_name):
    """
    Creates z-score and beta maps for a given GLM and regressor.

    This function creates the z-score and beta maps for a given GLM and regressor,
    then saves the maps to files and generates a report.

    Parameters:
    z_map_fname : str
        The name of the file to save the z-score map.
    beta_map_fname : str
        The name of the file to save the beta map.
    report_fname : str
        The name of the file to save the report.
    fmri_glm : nistats.regression.FirstLevelModel
        The GLM to use for creating the maps.
    regressor_name : str
        The name of the regressor for which to create the maps.

    Returns:
    None
    """
    # Check if regressor exists in the design matrix
    design_matrix = fmri_glm.design_matrices_[0]
    if regressor_name not in design_matrix.columns:
        raise ValueError(f"Regressor '{regressor_name}' not found in design matrix (condition did not occur in this session)")

    # Get betas
    beta_map = fmri_glm.compute_contrast(regressor_name, output_type="effect_size")
    beta_map.to_filename(beta_map_fname)

    # Get Z_map
    z_map = fmri_glm.compute_contrast(
        regressor_name, output_type="z_score", stat_type="F"
    )
    z_map.to_filename(z_map_fname)

    # Get report
    report = fmri_glm.generate_report(
        contrasts=[regressor_name],
        height_control=None
    )
    report.save_as_html(report_fname)


def select_events(run_events):
    """
    Selects and prepares event data for a given run.

    This function filters the events of a run based on a predefined list
    of conditions, prepares the event data for analysis, and returns it.

    Parameters:
    run_events : pandas.DataFrame
        The event data for a given run.

    Returns:
    annotation_events : pandas.DataFrame
        The prepared event data.
    """
    # Select events
    annotation_events = run_events[run_events["trial_type"].isin(CONDS_LIST)]
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]

    # replevel_events = run_events[run_events["trial_type"]=="gym-retro_game"] --- REMOVE ? MAKE FUNCTION ?
    # replevel_events["trial_type"] = replevel_events["level"]
    # replevel_events = replevel_events.replace({"trial_type": {"level-1": "lvl1", "level-4": "lvl4", "level-5": "lvl5"}})
    # replevel_events = replevel_events[["trial_type", "onset", "duration"]]
    # annotation_events = pd.concat((annotation_events, replevel_events), axis=0)
    return annotation_events


def load_run(fmri_fname, mask_fname, events_fname, use_low_level_confs=False):
    """
    Loads and prepares the data for a given run.

    This function loads the image data, event data, and mask for a given
    run, prepares the event data, and generates a clean design matrix.

    Parameters:
    fmri_fname : str
        The file name of the fMRI image data.
    mask_fname : str
        The file name of the mask.
    events_fname : str
        The file name of the event data.
    use_low_level_confs : bool
        Whether to include low-level confounds and button-press rate (default: False)

    Returns:
    design_matrix_clean : pandas.DataFrame
        The clean design matrix.
    fmri_img : nibabel.Nifti1Image
        The loaded fMRI image data.
    mask_resampled : nibabel.Nifti1Image
        The loaded mask.
    """
    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0], low_memory=False)
    annotation_events = select_events(run_events)

    # Load images
    fmri_img, mask_resampled = load_image_and_mask(fmri_fname, mask_fname)

    # Make design matrix
    design_matrix_clean = get_clean_matrix(
        fmri_fname, fmri_img, annotation_events, run_events, use_low_level_confs=use_low_level_confs
    )
    return design_matrix_clean, fmri_img, mask_resampled


def load_session(sub, ses, run_list, path_to_data, use_low_level_confs=False):
    """
    Loads and prepares the data for a given session.

    This function loads and prepares the image data, event data, and mask
    for each run in a given session.

    Parameters:
    sub : str
        The subject identifier.
    ses : str
        The session identifier.
    run_list : list
        The list of runs to load.
    path_to_data : str
        The path to the data.
    use_low_level_confs : bool
        Whether to include low-level confounds and button-press rate (default: False)

    Returns:
    fmri_imgs : list of nibabel.Nifti1Image
        The loaded fMRI image data for each run.
    design_matrices : list of pandas.DataFrame
        The clean design matrix for each run.
    mask_resampled : nibabel.Nifti1Image
        The loaded mask.
    anat_fname : str
        The file name of the anatomical image.
    """
    design_matrices = []
    fmri_imgs = []
    for run in run_list:
        fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
            sub, ses, run, path_to_data
        )
        # print(f"Loading : {fmri_fname}")
        design_matrix_clean, fmri_img, mask_resampled = load_run(
            fmri_fname, mask_fname, events_fname, use_low_level_confs=use_low_level_confs
        )
        design_matrices.append(design_matrix_clean)
        fmri_imgs.append(fmri_img)
    return fmri_imgs, design_matrices, mask_resampled, anat_fname


def remove_runs_without_target_regressor(
    regressor_names, fmri_imgs, trimmed_design_matrices
):
    """
    Removes runs that do not contain all target regressors.

    This function filters out the runs that do not contain all target
    regressors in their design matrix.

    Parameters:
    regressor_names : list of str
        The names of the target regressors.
    fmri_imgs : list of nibabel.Nifti1Image
        The fMRI images for each run.
    trimmed_design_matrices : list of pandas.DataFrame
        The design matrices for each run.

    Returns:
    images_copy : list of nibabel.Nifti1Image
        The fMRI images for the runs that contain all target regressors.
    dataframes_copy : list of pandas.DataFrame
        The design matrices for the runs that contain all target regressors.
    """
    images_copy = fmri_imgs.copy()
    dataframes_copy = trimmed_design_matrices.copy()
    for img, df in zip(fmri_imgs, trimmed_design_matrices):
        for reg in regressor_names:
            if reg not in df.columns:
                images_copy.remove(img)
                dataframes_copy.remove(df)

    return images_copy, dataframes_copy


def trim_design_matrices(design_matrices, regressor_name):
    """
    This function removes unwanted regressors from the design matrix for each run.

    Parameters
    ----------
    design_matrices : list
        A list of pandas DataFrames containing the design matrices for each run.
    regressor_name : str
        The name of the regressor that should not be removed.

    Returns
    -------
    list
        A list of trimmed design matrices for each run.
    """
    # Copy the list of conditions and levels to create a list of regressors to remove
    regressors_to_remove = CONDS_LIST.copy()
    # We don't want to remove the current regressor of interest, so remove it from the list of regressors to remove
    if not "lvl" in regressor_name:
        regressors_to_remove.remove(regressor_name)

    trimmed_design_matrices = []
    # For each design matrix, remove the unwanted regressors
    for design_matrix in design_matrices:
        trimmed_design_matrix = design_matrix
        for reg in regressors_to_remove:
            try:
                # Try to drop the current regressor from the design matrix
                trimmed_design_matrix = trimmed_design_matrix.drop(columns=reg)
            except Exception as e:
                # print(e)
                # print(f"Regressor {reg} might be missing ?")
                pass
        # Append the trimmed design matrix to the list of trimmed design matrices
        trimmed_design_matrices.append(trimmed_design_matrix)
    return trimmed_design_matrices


def make_or_load_glm(sub, ses, run_list, glm_regressors, glm_fname, mask_resampled_global=None, save_glm=False, use_low_level_confs=False):
    """
    Creates a General Linear Model (GLM) if it doesn't already exist, or loads it from disk if it does.

    This function uses the provided list of regressors to create the GLM. It first checks if the specified
    GLM file already exists. If it does, the function loads and returns the GLM. If not, the function creates
    the GLM.

    If image and design matrix data have been previously loaded and saved to global variables, the function
    uses this data to create the GLM, thereby avoiding reloading all the data. Once the GLM has been created,
    it's dumped to the specified file for future use.

    Parameters:
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    run_list : list
        List of runs to consider.
    glm_regressors : list
        List of regressors to consider when creating the GLM.
    glm_fname : str
        The name of the file to which the GLM will be dumped, or from which it will be loaded.
    mask_resampled_global : nibabel.Nifti1Image, optional
        Preloaded mask to avoid reloading.

    Returns:
    fmri_glm : nistats.regression.FirstLevelModel
        The GLM, either loaded from disk or newly created.
    mask_resampled : nibabel.Nifti1Image
        The mask used for the GLM.
    """
    if save_glm and os.path.exists(glm_fname):
        # Load existing GLM
        with open(glm_fname, "rb") as f:
            # print(f"GLM found, loading : {glm_fname}")
            fmri_glm = pickle.load(f)
            # print("Loaded.")
        mask_resampled = mask_resampled_global
    else:
        # Compute GLM fresh (either save_glm=False or GLM doesn't exist)
        fmri_imgs, design_matrices, mask_resampled, anat_fname = load_session(
            sub, ses, run_list, path_to_data, use_low_level_confs=use_low_level_confs
        )

        # print(f"Computing GLM : {glm_fname}")
        # print(glm_regressors)
        trimmed_design_matrices = trim_design_matrices(
            design_matrices, glm_regressors[0]
        )
        fmri_imgs, trimmed_design_matrices = remove_runs_without_target_regressor(
            glm_regressors, fmri_imgs, trimmed_design_matrices
        )
        if len(glm_regressors) == 2:
            for dm in trimmed_design_matrices:
                dm.eval(
                    f"{glm_regressors[0]}X{glm_regressors[1]} = {glm_regressors[0]} * {glm_regressors[1]}",
                    inplace=True,
                )

        # Use provided mask if available, otherwise use loaded one
        mask_to_use = mask_resampled_global if mask_resampled_global is not None else mask_resampled

        fmri_glm = make_and_fit_glm(fmri_imgs, trimmed_design_matrices, mask_to_use)

        if save_glm:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(glm_fname), exist_ok=True)

            with open(glm_fname, "wb") as f:
                pickle.dump(fmri_glm, f, protocol=4)

    return fmri_glm, mask_resampled


def process_ses(sub, ses, path_to_data, save_glm=False, use_low_level_confs=False, logger=None):
    """
    Process an fMRI session for a given subject and session.
    It runs General Linear Models (GLM) for different regressors, creating them if they don't already exist.
    Computes GLMs incrementally for 1 run, 2 runs, 3 runs, etc.

    Parameters:
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    path_to_data : str
        Path to the data directory.
    save_glm : bool
        Whether to save GLM objects to disk (default: False)
    use_low_level_confs : bool
        Whether to include low-level confounds and button-press rate (default: False)
    logger : GLMLogger, optional
        Logger instance for tracking processing.

    Returns:
    None
    """

    def process_regressor(regressor_name, run_list_subset, n_runs_label, lvl=None):
        """
        Process a single regressor with a specific subset of runs.

        Parameters:
        regressor_name : str
            Name of the regressor to process
        run_list_subset : list
            List of runs to include in this GLM
        n_runs_label : int or None
            Number of runs being processed (for directory naming). None means all runs.
        lvl : str, optional
            Level interaction term
        """
        if lvl is None:
            glm_regressors = [regressor_name]
            regressor_output_name = regressor_name
        else:
            glm_regressors = [regressor_name] + [lvl]
            regressor_output_name = f"{regressor_name}X{lvl}"

        if logger:
            logger.debug(f"Processing regressor: {regressor_output_name} with {len(run_list_subset)} runs")
        else:
            print(f"Simple model of : {regressor_output_name} with {len(run_list_subset)} runs")

        glm_fname, z_map_fname, beta_map_fname, report_fname = get_output_names(
            sub, ses, regressor_output_name, n_runs=n_runs_label, use_low_level_confs=use_low_level_confs
        )

        if not (os.path.exists(z_map_fname)):
            try:
                if logger:
                    logger.log_computation_start(f"{regressor_output_name}", z_map_fname)

                fmri_glm, _ = make_or_load_glm(sub, ses, run_list_subset, glm_regressors, glm_fname, save_glm=save_glm, use_low_level_confs=use_low_level_confs)
                make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_output_name)

                if logger:
                    logger.log_computation_success(f"{regressor_output_name}", z_map_fname)
            except Exception as e:
                if logger:
                    logger.log_computation_error(f"{regressor_output_name}", e)
                else:
                    print(f"Error processing {regressor_output_name}: {e}")
        else:
            if logger:
                logger.log_computation_skip(f"{regressor_output_name}", z_map_fname)
            else:
                print(f"Z map found, skipping : {z_map_fname}")

    # Get list of runs for this session
    import re
    ses_fpath = op.join(path_to_data, "shinobi.fmriprep", sub, ses, "func")
    ses_files = os.listdir(ses_fpath)
    run_files = [
        x
        for x in ses_files
        if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in x
    ]
    # Extract run numbers from BIDS filenames using regex
    run_list = []
    for fname in run_files:
        match = re.search(r'run-(\d+)', fname)
        if match:
            run_list.append(match.group(1))  # Keep as string for consistency
    run_list = sorted(run_list)

    if logger:
        logger.log_run_list(run_list)
    else:
        print(f"Found {len(run_list)} runs for {sub} {ses}: {run_list}")

    # Process incrementally: 1 run, 2 runs, 3 runs, etc.
    for n_runs in range(1, len(run_list) + 1):
        run_list_subset = run_list[:n_runs]

        # Determine the directory label
        # If using all runs, store in ses-level; otherwise in ses-level_Nrun
        n_runs_label = None if n_runs == len(run_list) else n_runs

        if not logger:
            print(f"\n{'='*60}")
            print(f"Processing {sub} {ses} with {n_runs} run(s): {run_list_subset}")
            print(f"{'='*60}\n")

        # Make a GLM with each regressor separately (simple models)
        for regressor_name in CONDS_LIST + LEVELS:
            process_regressor(regressor_name, run_list_subset, n_runs_label)

        # Still simple models but split by level (interaction annotation X level)
        for lvl in LEVELS:
            for regressor_name in CONDS_LIST:
                process_regressor(regressor_name, run_list_subset, n_runs_label, lvl)

    return


def main():
    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="GLM_session",
        subject=sub,
        session=ses,
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        process_ses(sub, ses, path_to_data, save_glm=args.save_glm, use_low_level_confs=args.low_level_confs, logger=logger)
    finally:
        # Always close logger to print summary
        logger.close()


if __name__ == "__main__":
    figures_path = (
        shinobi_behav.FIG_PATH
    )  #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    CONDS_LIST = ["HIT", "JUMP", "DOWN", "LEFT", "RIGHT", "UP", "Kill", "HealthLoss"]
    LEVELS = []  # ["lvl1", "lvl4", "lvl5"]
    additional_contrasts = ["HIT+JUMP", "RIGHT+LEFT+DOWN"]
    sub = args.subject
    ses = args.session
    t_r = 1.49
    hrf_model = "spm"
    # Log job info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {ses}")
    print(f"Writing processed data in : {path_to_data}")
    print(f"Writing reports in : {figures_path}")
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
