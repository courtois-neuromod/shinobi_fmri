import os
import os.path as op
import pandas as pd
import warnings
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import get_scrub_regressor
from shinobi_fmri.utils.logger import ShinobiLogger
import numpy as np
import pdb
import argparse
import nilearn
import config

# Suppress informational warnings
warnings.filterwarnings('ignore', message='.*imgs are being resampled to the mask_img resolution.*')
warnings.filterwarnings('ignore', message='.*Mean values of 0 observed.*')
warnings.filterwarnings('ignore', message='.*design matrices are supplied.*')
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.image import clean_img
from nilearn.reporting import get_clusters_table
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.signal import clean
import nibabel as nib
import logging
import pickle
from nilearn.plotting import plot_img_on_surf, plot_stat_map
import glob
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-06",
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-ses",
    "--session",
    default="ses-010",
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


def get_filenames(sub, ses, run, path_to_data):
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
    mask_resampled = image.resample_img(
        mask_fname,
        target_affine=target_affine,
        target_shape=fmri_img.get_fdata().shape[:3],
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
        if button not in frame_events.columns:
            print(f"Warning: Button {button} not in frame_events columns")
            continue
        
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

        # print(f"Button {button}: {button_states.sum()} frames with button held")

        # Detect transitions: False->True = press, True->False = release
        presses = np.zeros(len(button_states), dtype=bool)
        releases = np.zeros(len(button_states), dtype=bool)

        for i in range(1, len(button_states)):
            if not button_states[i-1] and button_states[i]:
                presses[i] = True
            elif button_states[i-1] and not button_states[i]:
                releases[i] = True

        # print(f"Button {button}: {presses.sum()} presses, {releases.sum()} releases detected")
        
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

    design_matrix_full = get_scrub_regressor(run_events, design_matrix_raw)
    return design_matrix_full

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
    # Check if regressor exists in the design matrix
    design_matrix = fmri_glm.design_matrices_[0]
    if regressor_name not in design_matrix.columns:
        raise ValueError(f"Regressor '{regressor_name}' not found in design matrix (condition did not occur in this run)")

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
    return z_map


def load_run(fmri_fname, mask_fname, events_fname, use_low_level_confs=False):
    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0], low_memory=False)
    # Select events
    annotation_events = run_events[run_events["trial_type"].isin(CONDS_LIST)]
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]

    # Load images
    fmri_img, mask_resampled = load_image_and_mask(fmri_fname, mask_fname)

    # Make design matrix
    design_matrix_clean = get_clean_matrix(
        fmri_fname, fmri_img, annotation_events, run_events, use_low_level_confs=use_low_level_confs
    )
    return design_matrix_clean, fmri_img, mask_resampled


def load_session(sub, ses, run_list, path_to_data):
    design_matrices = []
    fmri_imgs = []
    for run in run_list:
        fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
            sub, ses, run, path_to_data
        )
        print(f"Loading : {fmri_fname}")
        design_matrix_clean, fmri_img, mask_resampled = load_run(
            fmri_fname, mask_fname, events_fname
        )
        design_matrices.append(design_matrix_clean)
        fmri_imgs.append(fmri_img)
    return fmri_imgs, design_matrices, mask_resampled, anat_fname


def process_run(sub, ses, run, path_to_data, save_glm=False, use_low_level_confs=False, logger=None):
    # Determine output directory based on whether low-level confounds are used
    # Note: low-level confounds info is in the directory name (processed vs processed_low-level)
    output_dir = "processed_low-level" if use_low_level_confs else "processed"

    # Format run number to be BIDS-compliant (e.g., "2" -> "run-02")
    run_formatted = f"run-{int(run):02d}"

    # Compute GLM for each condition separately
    for regressor_name in CONDS_LIST:
        try:
            # BIDS-compliant directory structure: processed/sub-XX/ses-YY/func/
            func_dir = op.join(path_to_data, output_dir, sub, ses, "func")
            os.makedirs(func_dir, exist_ok=True)

            # BIDS-compliant filename: sub-XX_ses-YY_task-shinobi_run-XX_contrast-CONDITION_stat-z.nii.gz
            base_name = f"{sub}_{ses}_task-shinobi_{run_formatted}_contrast-{regressor_name}"
            z_map_fname = op.join(func_dir, f"{base_name}_stat-z.nii.gz")

            if os.path.exists(z_map_fname):
                if logger:
                    logger.log_computation_skip(regressor_name, z_map_fname)
                # else:
                #     print(f"Z map found, skipping : {z_map_fname}")
                continue

            # GLM file (if saving)
            glm_fname = op.join(func_dir, f"{base_name}_glm.pkl")
            if save_glm:
                os.makedirs(func_dir, exist_ok=True)

            if save_glm and os.path.exists(glm_fname):
                # Load existing GLM
                with open(glm_fname, "rb") as f:
                    if logger:
                        logger.info(f"GLM found, loading : {glm_fname}")
                    # else:
                    #     print(f"GLM found, loading : {glm_fname}")
                    fmri_glm = pickle.load(f)
                    if logger:
                        logger.info("Loaded.")
                    # else:
                    #     print("Loaded.")
            else:
                # Compute GLM fresh (either save_glm=False or GLM doesn't exist)
                fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
                    sub, ses, run, path_to_data
                )
                if logger:
                    logger.info(f"Loading : {fmri_fname}")
                # else:
                #     print(f"Loading : {fmri_fname}")
                design_matrix_clean, fmri_img, mask_resampled = load_run(
                    fmri_fname, mask_fname, events_fname, use_low_level_confs=use_low_level_confs
                )
                design_matrices = [design_matrix_clean]
                fmri_imgs = [fmri_img]
                # Trim the design matrices from unwanted regressors
                regressors_to_remove = CONDS_LIST.copy()
                regressors_to_remove.remove(regressor_name)
                trimmed_design_matrices = []
                for design_matrix in design_matrices:
                    trimmed_design_matrix = design_matrix
                    for reg in regressors_to_remove:
                        try:
                            trimmed_design_matrix = trimmed_design_matrix.drop(
                                columns=reg
                            )
                        except Exception as e:
                            if logger:
                                logger.warning(f"{e}\nRegressor {reg} might be missing ?")
                            # else:
                            #     print(e)
                            #     print(f"Regressor {reg} might be missing ?")
                    trimmed_design_matrices.append(trimmed_design_matrix)

                fmri_glm = make_and_fit_glm(
                    fmri_imgs, trimmed_design_matrices, mask_resampled
                )
                if save_glm:
                    with open(glm_fname, "wb") as f:
                        pickle.dump(fmri_glm, f, protocol=4)

            # Compute contrast and z-map
            if logger:
                logger.log_computation_start(regressor_name, z_map_fname)
            # else:
            #     print(f"Computing : {z_map_fname}")

            # Report filename (keeping reports in figures_path for now)
            report_dir = op.join(figures_path, "run-level", regressor_name, "report")
            os.makedirs(report_dir, exist_ok=True)
            report_fname = op.join(report_dir, f"{base_name}_report.html")

            # Beta map filename
            beta_map_fname = op.join(func_dir, f"{base_name}_stat-beta.nii.gz")

            z_map = make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_name)
            if logger:
                logger.log_computation_success(regressor_name, z_map_fname)
        except Exception as e:
            if logger:
                logger.log_computation_error(regressor_name, e)
            else:
                print(e)
    return


def process_ses(sub, ses, path_to_data, save_glm=False, use_low_level_confs=False, logger=None):
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

    for run in sorted(run_list):
        if logger:
            logger.info(f"Run : {run}")
        else:
            print(f"Run : {run}")
        process_run(sub, ses, run, path_to_data, save_glm=save_glm, use_low_level_confs=use_low_level_confs, logger=logger)
    return


def main(logger=None):
    fmri_glm = process_ses(sub, ses, path_to_data, save_glm=args.save_glm, use_low_level_confs=args.low_level_confs, logger=logger)


if __name__ == "__main__":
    figures_path = (
        config.FIG_PATH
    )  #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = config.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    CONDS_LIST = [
        "HIT",
        "JUMP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "UP",
        "Kill",
        "HealthGain",
        "HealthLoss",
    ]
    additional_contrasts = ["HIT+JUMP", "RIGHT+LEFT+DOWN"]
    sub = args.subject
    ses = args.session
    t_r = 1.49
    hrf_model = "spm"
    # Log job info
    # Log job info
    # log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {ses}")
    # print(f"Writing processed data in : {path_to_data}")
    # print(f"Writing reports in : {figures_path}")
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="GLM_run",
        subject=sub,
        session=ses,
        log_dir=args.log_dir,
        verbosity=log_level
    )
    
    logger.info(f"Writing processed data in : {path_to_data}")
    logger.info(f"Writing reports in : {figures_path}")

    try:
        main(logger=logger)
    finally:
        logger.close()
