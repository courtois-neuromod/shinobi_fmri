import os
import os.path as op
import pandas as pd
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import get_scrub_regressor
import numpy as np
import pdb
import argparse
import nilearn
import shinobi_behav
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.image import clean_img
from nilearn.reporting import get_clusters_table
from nilearn import input_data
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
        Run number
    path_to_data : str
        Path to the data folder
    Returns
    -------
    tuple
        Tuple containing file names of fMRI, anatomy, and annotated events
    """
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
        f"{sub}_{ses}_task-shinobi_run-0{run}_desc-annotated_events.tsv",
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
        print("Warning: No frame events found in run_events")
        return confounds
    
    print(f"Found {len(frame_events)} frame events")
    
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
        
        print(f"Button {button}: {button_states.sum()} frames with button held")
        
        # Detect transitions: False->True = press, True->False = release
        presses = np.zeros(len(button_states), dtype=bool)
        releases = np.zeros(len(button_states), dtype=bool)
        
        for i in range(1, len(button_states)):
            if not button_states[i-1] and button_states[i]:
                presses[i] = True
            elif button_states[i-1] and not button_states[i]:
                releases[i] = True
        
        print(f"Button {button}: {presses.sum()} presses, {releases.sum()} releases detected")
        
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
    Add psychophysics confounds to the confounds dataframe
    """
    n_volumes = len(confounds[0])
    ppc_data = {}  # Dictionary to store accumulated data for each key

    for idx, row in run_events.iterrows():
        if row["trial_type"] != "gym-retro_game":
            continue

        ppc_fname = op.join(
            path_to_data, "shinobi", row["stim_file"].replace(".bk2", "_confs.npy")
        )

        if not os.path.exists(ppc_fname):
            continue

        ppc_rep = np.load(ppc_fname, allow_pickle=True).item()
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

    return confounds


def get_clean_matrix(fmri_fname, fmri_img, annotation_events, run_events):
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


def make_z_map(z_map_fname, report_fname, fmri_glm, regressor_name):
    # Get betas
    beta_map = fmri_glm.compute_contrast(regressor_name, output_type="effect_size")
    os.makedirs(
        op.join(path_to_data, "processed", "beta_maps", "run-level", regressor_name),
        exist_ok=True,
    )
    beta_map.to_filename(z_map_fname.replace("z_maps", "beta_maps"))
    # Get Z_map
    z_map = fmri_glm.compute_contrast(
        regressor_name, output_type="z_score", stat_type="F"
    )
    os.makedirs(
        op.join(path_to_data, "processed", "z_maps", "run-level", regressor_name),
        exist_ok=True,
    )
    z_map.to_filename(z_map_fname)

    # Get report
    os.makedirs(
        op.join(figures_path, "run-level", regressor_name, "report"), exist_ok=True
    )
    report = fmri_glm.generate_report(contrasts=[regressor_name])
    report.save_as_html(report_fname)
    return z_map


def load_run(fmri_fname, mask_fname, events_fname):
    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0])
    # Select events
    annotation_events = run_events[run_events["trial_type"].isin(CONDS_LIST)]
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]

    # Load images
    fmri_img, mask_resampled = load_image_and_mask(fmri_fname, mask_fname)

    # Make design matrix
    design_matrix_clean = get_clean_matrix(
        fmri_fname, fmri_img, annotation_events, run_events
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


def process_run(sub, ses, run, path_to_data):

    # Full model first
    glm_fname = op.join(
        path_to_data,
        "processed",
        "glm",
        "run-level",
        sub,
        f"{sub}_{ses}_{run}_fullmodel_fitted_glm.pkl",
    )
    os.makedirs(
        op.join(path_to_data, "processed", "glm", "run-level", sub), exist_ok=True
    )
    if not (os.path.exists(glm_fname)):
        fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
            sub, ses, run, path_to_data
        )
        print(f"Loading : {fmri_fname}")
        design_matrix_clean, fmri_img, mask_resampled = load_run(
            fmri_fname, mask_fname, events_fname
        )
        fmri_glm = make_and_fit_glm(fmri_img, design_matrix_clean, mask_resampled)
        with open(glm_fname, "wb") as f:
            pickle.dump(fmri_glm, f, protocol=4)
    else:
        with open(glm_fname, "rb") as f:
            print(f"GLM found, loading : {glm_fname}")
            fmri_glm = pickle.load(f)
            print("Loaded.")
        # Compute all contrasts
    for regressor_name in CONDS_LIST + additional_contrasts:
        try:
            z_map_fname = op.join(
                path_to_data,
                "processed",
                "z_maps",
                "run-level",
                regressor_name,
                f"{sub}_{ses}_{run}_fullmodel_{regressor_name}.nii.gz",
            )
            os.makedirs(
                op.join(
                    path_to_data, "processed", "z_maps", "run-level", regressor_name
                ),
                exist_ok=True,
            )
            if not (os.path.exists(z_map_fname)):
                print(f"Z map not found, computing : {z_map_fname}")
                report_fname = op.join(
                    figures_path,
                    "run-level",
                    regressor_name,
                    "report",
                    f"{sub}_{ses}_{run}_fullmodel_{regressor_name}_report.html",
                )
                os.makedirs(
                    op.join(figures_path, "run-level", regressor_name, "report"),
                    exist_ok=True,
                )
                z_map = make_z_map(z_map_fname, report_fname, fmri_glm, regressor_name)
            else:
                print(f"Z map found, skipping : {z_map_fname}")
        except Exception as e:
            print(e)

    # Intermediate model
    for regressor_name in additional_contrasts:
        try:
            glm_fname = op.join(
                path_to_data,
                "processed",
                "glm",
                "run-level",
                sub,
                f"{sub}_{ses}_{run}_intermediatemodel_fitted_glm.pkl",
            )
            os.makedirs(
                op.join(path_to_data, "processed", "glm", "run-level", sub),
                exist_ok=True,
            )
            if not (os.path.exists(glm_fname)):
                print(f"GLM not found, computing : {glm_fname}")
                fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
                    sub, ses, run, path_to_data
                )
                print(f"Loading : {fmri_fname}")
                design_matrix_clean, fmri_img, mask_resampled = load_run(
                    fmri_fname, mask_fname, events_fname
                )

                # Trim the design matrices from unwanted regressors
                regressors_to_remove = CONDS_LIST.copy()
                for toremove in ["HIT", "JUMP", "LEFT", "RIGHT", "DOWN"]:
                    regressors_to_remove.remove(toremove)

                trimmed_design_matrix = design_matrix_clean
                for reg in regressors_to_remove:
                    try:
                        trimmed_design_matrix = trimmed_design_matrix.drop(columns=reg)
                    except Exception as e:
                        print(e)
                        print(f"Regressor {reg} might be missing ?")

                fmri_glm = make_and_fit_glm(
                    fmri_img, trimmed_design_matrix, mask_resampled
                )
                with open(glm_fname, "wb") as f:
                    pickle.dump(fmri_glm, f, protocol=4)
            else:
                with open(glm_fname, "rb") as f:
                    print(f"GLM found, loading : {glm_fname}")
                    fmri_glm = pickle.load(f)
                    print("Loaded.")

            # Compute contrast
            z_map_fname = op.join(
                path_to_data,
                "processed",
                "z_maps",
                "run-level",
                regressor_name,
                f"{sub}_{ses}_{run}_intermediatemodel_{regressor_name}.nii.gz",
            )
            os.makedirs(
                op.join(path_to_data, "processed", "z_maps", "run-level"), exist_ok=True
            )
            if not (os.path.exists(z_map_fname)):
                print(f"Z map not found, computing : {z_map_fname}")
                report_fname = op.join(
                    figures_path,
                    "run-level",
                    regressor_name,
                    "report",
                    f"{sub}_{ses}_{run}_intermediatemodel_{regressor_name}_report.html",
                )
                os.makedirs(
                    op.join(figures_path, "run-level", regressor_name, "report"),
                    exist_ok=True,
                )
                z_map = make_z_map(z_map_fname, report_fname, fmri_glm, regressor_name)
            else:
                print(f"Z map found, skipping : {z_map_fname}")
        except Exception as e:
            print(e)

    # Simple model
    for regressor_name in CONDS_LIST:
        try:
            glm_fname = op.join(
                path_to_data,
                "processed",
                "glm",
                "run-level",
                sub,
                f"{sub}_{ses}_{run}_{regressor_name}_simplemodel_fitted_glm.pkl",
            )
            os.makedirs(
                op.join(path_to_data, "processed", "glm", "run-level", sub),
                exist_ok=True,
            )
            if not (os.path.exists(glm_fname)):
                print(f"GLM not found, computing : {glm_fname}")
                fmri_fname, anat_fname, events_fname, mask_fname = get_filenames(
                    sub, ses, run, path_to_data
                )
                print(f"Loading : {fmri_fname}")
                design_matrix_clean, fmri_img, mask_resampled = load_run(
                    fmri_fname, mask_fname, events_fname
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
                            print(e)
                            print(f"Regressor {reg} might be missing ?")
                    trimmed_design_matrices.append(trimmed_design_matrix)

                fmri_glm = make_and_fit_glm(
                    fmri_imgs, trimmed_design_matrices, mask_resampled
                )
                with open(glm_fname, "wb") as f:
                    pickle.dump(fmri_glm, f, protocol=4)
            else:
                with open(glm_fname, "rb") as f:
                    print(f"GLM found, loading : {glm_fname}")
                    fmri_glm = pickle.load(f)
                    print("Loaded.")

            # Compute contrast
            z_map_fname = op.join(
                path_to_data,
                "processed",
                "z_maps",
                "run-level",
                regressor_name,
                f"{sub}_{ses}_{run}_simplemodel_{regressor_name}.nii.gz",
            )
            os.makedirs(
                op.join(
                    path_to_data, "processed", "z_maps", "run-level", regressor_name
                ),
                exist_ok=True,
            )
            if not (os.path.exists(z_map_fname)):
                print(f"Z map not found, computing : {z_map_fname}")
                report_fname = op.join(
                    figures_path,
                    "run-level",
                    regressor_name,
                    "report",
                    f"{sub}_{ses}_{run}_simplemodel_{regressor_name}_report.html",
                )
                os.makedirs(
                    op.join(figures_path, "run-level", regressor_name, "report"),
                    exist_ok=True,
                )
                z_map = make_z_map(z_map_fname, report_fname, fmri_glm, regressor_name)
            else:
                print(f"Z map found, skipping : {z_map_fname}")
        except Exception as e:
            print(e)
    return


def process_ses(sub, ses, path_to_data):
    ses_fpath = op.join(path_to_data, "shinobi.fmriprep", sub, ses, "func")
    ses_files = os.listdir(ses_fpath)
    run_files = [
        x
        for x in ses_files
        if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in x
    ]
    run_list = [x[32] for x in run_files]
    for run in sorted(run_list):
        print(f"Run : {run}")
        process_run(sub, ses, run, path_to_data)
    return


def main():
    fmri_glm = process_ses(sub, ses, path_to_data)


if __name__ == "__main__":
    figures_path = (
        shinobi_behav.FIG_PATH
    )  #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
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
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {ses}")
    print(f"Writing processed data in : {path_to_data}")
    print(f"Writing reports in : {figures_path}")
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
