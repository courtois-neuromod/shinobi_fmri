
import os
import os.path as op
import pandas as pd
import numpy as np
import logging
import nibabel as nib
from nilearn import image
from nilearn.image import clean_img
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import cluster_level_inference
from shinobi_fmri.annotations.annotations import get_scrub_regressor
from shinobi_fmri import config

# Constants
TR = 1.49
HRF_MODEL = "spm"

def get_filenames(sub, ses, run, path_to_data):
    """
    Returns file names for fMRI, anatomy and annotation events.
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

def load_image_and_mask(fmri_fname, mask_fname, t_r=TR):
    """
    Load and clean 4d image and resample anat mask
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

def downsample_to_TR(signal, fs=60.0, TR=TR):
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

def add_button_press_confounds(confounds, run_events, t_r=TR):
    """
    Add button press/release event regressors to the confounds dataframe.
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
    confounds[0]['button_count_press'] = count_press
    #confounds[0]['button_count_release'] = count_release
    
    return confounds

def add_psychophysics_confounds(confounds, run_events, path_to_data, t_r=TR):
    """
    Add low-level features confounds to the confounds dataframe
    """
    n_volumes = len(confounds[0])
    ppc_data = {}  # Dictionary to store accumulated data for each key
    n_invalid_onsets = 0

    for idx, row in run_events.iterrows():
        if row["trial_type"] != "gym-retro_game":
            continue

        if pd.isna(row["onset"]):
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

def get_clean_matrix(fmri_fname, fmri_img, annotation_events, run_events, path_to_data, use_low_level_confs=False, t_r=TR, hrf_model=HRF_MODEL):
    """
    Load confounds, create design matrix and return a cleaned matrix
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

def make_and_fit_glm(fmri_imgs, design_matrices, mask_resampled, t_r=TR, hrf_model=HRF_MODEL):
    """
    Perform GLM analysis
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

def make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_name, cluster_thresh=2.3, alpha=0.05):
    """
    Creates z-score and beta maps for a given GLM and regressor.
    Also computes a cluster-corrected z-map.
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
    if cluster_thresh is not None:
        try:
            corrected_map = cluster_level_inference(z_map, threshold=cluster_thresh, alpha=alpha)
            # BIDS-compliant naming: insert 'desc-corrected' before 'stat-z'
            # Assumes filename ends with _stat-z.nii.gz
            corrected_fname = z_map_fname.replace('_stat-z.nii.gz', '_desc-corrected_stat-z.nii.gz')
            corrected_map.to_filename(corrected_fname)
        except Exception as e:
            print(f"Warning: Failed to compute cluster correction for {regressor_name}: {e}")

    # Get report
    report = fmri_glm.generate_report(
        contrasts=[regressor_name],
        height_control=None
    )
    report.save_as_html(report_fname)
    return z_map

def select_events(run_events, conditions_list):
    """
    Selects events
    """
    annotation_events = run_events[run_events["trial_type"].isin(conditions_list)]
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]
    return annotation_events

def load_run(fmri_fname, mask_fname, events_fname, path_to_data, conditions_list, use_low_level_confs=False):
    """
    Loads and prepares the data for a given run.
    """
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
