import os
import os.path as op
import pandas as pd
from nilearn import image
from shinobi_fmri.annotations.annotations import get_scrub_regressor
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

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        Run number
    path_to_data : strrun: str
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


def get_output_names(sub, ses, regressor_output_name):
    """
    Constructs and returns the file paths for the GLM, z-map, and report files.

    This function constructs the file paths for the GLM (general linear model),
    z-map, and report files for a given subject, session, and regressor, using a predefined
    directory structure and file naming convention.

    Parameters:
    sub : str
        The subject identifier.
    ses : str
        The session identifier.
    regressor_output_name : str
        The name of the regressor, used in the naming of the output files.

    Returns:
    glm_fname : str
        The file path for the GLM file.
    z_map_fname : str
        The file path for the z-map file.
    report_fname : str
        The file path for the report file.
    """
    glm_fname = op.join(
        shinobi_behav.DATA_PATH,
        "processed",
        "glm",
        "ses-level",
        f"{sub}_{ses}_{regressor_output_name}_simplemodel_fitted_glm.pkl",
    )

    z_map_fname = op.join(
        shinobi_behav.DATA_PATH,
        "processed",
        "z_maps",
        "ses-level",
        regressor_output_name,
        f"{sub}_{ses}_simplemodel_{regressor_output_name}.nii.gz",
    )

    report_fname = op.join(
        shinobi_behav.FIG_PATH,
        "ses-level",
        regressor_output_name,
        "report",
        f"{sub}_{ses}_simplemodel_{regressor_output_name}_report.html",
    )
    return glm_fname, z_map_fname, report_fname


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
        mask_fname, target_affine=target_affine, target_shape=target_shape
    )
    return fmri_img, mask_resampled


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
    confounds = add_psychophysiological_confounds(confounds, run_events)

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


def add_psychophysiological_confounds(confounds, run_events):
    """
    Add psychophysiological confounds to the confounds dataframe
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
    """
    Creates or loads a z-score map for a given GLM and regressor.

    This function creates the z-score map for a given GLM and regressor,
    then saves the map to a file and generates a report. If the z-score
    map file already exists, it skips the creation process.

    Parameters:
    z_map_fname : str
        The name of the file to save the z-score map.
    report_fname : str
        The name of the file to save the report.
    fmri_glm : nistats.regression.FirstLevelModel
        The GLM to use for creating the z-score map.
    regressor_name : str
        The name of the regressor for which to create the z-score map.

    Returns:
    None
    """
    if not (os.path.exists(z_map_fname)):
        print(f"Z map not found, computing : {z_map_fname}")

        # Get betas
        beta_map = fmri_glm.compute_contrast(regressor_name, output_type="effect_size")
        os.makedirs(
            op.join(
                path_to_data, "processed", "beta_maps", "ses-level", regressor_name
            ),
            exist_ok=True,
        )
        beta_map.to_filename(z_map_fname.replace("z_maps", "beta_maps"))
        # Get Z_map
        z_map = fmri_glm.compute_contrast(
            regressor_name, output_type="z_score", stat_type="F"
        )
        os.makedirs(
            op.join(path_to_data, "processed", "z_maps", "ses-level", regressor_name),
            exist_ok=True,
        )
        z_map.to_filename(z_map_fname)

        # Get report
        os.makedirs(
            op.join(figures_path, "ses-level", regressor_name, "report"), exist_ok=True
        )
        report = fmri_glm.generate_report(contrasts=[regressor_name])
        report.save_as_html(report_fname)
    else:
        print(f"Z map found, skipping : {z_map_fname}")
    # return z_map


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


def load_run(fmri_fname, mask_fname, events_fname):
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

    Returns:
    design_matrix_clean : pandas.DataFrame
        The clean design matrix.
    fmri_img : nibabel.Nifti1Image
        The loaded fMRI image data.
    mask_resampled : nibabel.Nifti1Image
        The loaded mask.
    """
    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0])
    annotation_events = select_events(run_events)

    # Load images
    fmri_img, mask_resampled = load_image_and_mask(fmri_fname, mask_fname)

    # Make design matrix
    design_matrix_clean = get_clean_matrix(
        fmri_fname, fmri_img, annotation_events, run_events
    )
    return design_matrix_clean, fmri_img, mask_resampled


def load_session(sub, ses, run_list, path_to_data):
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
        print(f"Loading : {fmri_fname}")
        design_matrix_clean, fmri_img, mask_resampled = load_run(
            fmri_fname, mask_fname, events_fname
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
                print(e)
                print(f"Regressor {reg} might be missing ?")
        # Append the trimmed design matrix to the list of trimmed design matrices
        trimmed_design_matrices.append(trimmed_design_matrix)
    return trimmed_design_matrices


def make_or_load_glm(sub, ses, run_list, glm_regressors, glm_fname):
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

    Returns:
    fmri_glm : nistats.regression.FirstLevelModel
        The GLM, either loaded from disk or newly created.
    """
    global fmri_imgs_copy, design_matrices_copy, mask_resampled
    if not (os.path.exists(glm_fname)):
        # Avoid reloading all the data if it is already loaded
        if fmri_imgs_copy is None:
            fmri_imgs, design_matrices, mask_resampled, anat_fname = load_session(
                sub, ses, run_list, path_to_data
            )
            fmri_imgs_copy = fmri_imgs.copy()
            design_matrices_copy = design_matrices.copy()
        else:
            fmri_imgs = fmri_imgs_copy.copy()
            design_matrices = design_matrices_copy.copy()

        print(f"GLM not found, computing : {glm_fname}")
        print(glm_regressors)
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
        fmri_glm = make_and_fit_glm(fmri_imgs, trimmed_design_matrices, mask_resampled)
        with open(glm_fname, "wb") as f:
            pickle.dump(fmri_glm, f, protocol=4)
    else:
        with open(glm_fname, "rb") as f:
            print(f"GLM found, loading : {glm_fname}")
            fmri_glm = pickle.load(f)
            print("Loaded.")
    return fmri_glm


def process_ses(sub, ses, path_to_data):
    """
    Process an fMRI session for a given subject and session.
    It runs General Linear Models (GLM) for different regressors, creating them if they don't already exist.

    Parameters:
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    path_to_data : str
        Path to the data directory.

    Returns:
    None
    """
    global fmri_imgs_copy, design_matrices_copy
    fmri_imgs_copy = None
    design_matrices_copy = None

    def process_regressor(regressor_name, lvl=None):
        global fmri_imgs_copy, design_matrices_copy
        if lvl is None:
            glm_regressors = [regressor_name]
            regressor_output_name = regressor_name
        else:
            glm_regressors = [regressor_name] + [lvl]
            regressor_output_name = f"{regressor_name}X{lvl}"
        glm_regressors = [regressor_name] if lvl is None else [regressor_name] + [lvl]

        print(f"Simple model of : {regressor_output_name}")
        glm_fname, z_map_fname, report_fname = get_output_names(
            sub, ses, regressor_output_name
        )
        if not (os.path.exists(z_map_fname)):
            fmri_glm = make_or_load_glm(sub, ses, run_list, glm_regressors, glm_fname)
            make_z_map(z_map_fname, report_fname, fmri_glm, regressor_output_name)
        else:
            print(f"Z map found, skipping : {z_map_fname}")

    ses_fpath = op.join(path_to_data, "shinobi.fmriprep", sub, ses, "func")
    ses_files = os.listdir(ses_fpath)
    run_files = [
        x
        for x in ses_files
        if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in x
    ]
    run_list = [x[32] for x in run_files]

    # Make a GLM with each regressor separately (simple models)
    for regressor_name in CONDS_LIST + LEVELS:
        try:
            process_regressor(regressor_name)
        except Exception as e:
            print(e)

    # Still simple models but split by level (interaction annotation X level) --- TO REMOVE ??
    for lvl in LEVELS:
        for regressor_name in CONDS_LIST:
            try:
                process_regressor(regressor_name, lvl)
            except Exception as e:
                print(e)
    return


def main():
    # Make folders if needed
    os.makedirs(op.join(path_to_data, "processed", "glm", "ses-level"), exist_ok=True)
    os.makedirs(
        op.join(path_to_data, "processed", "z_maps", "ses-level"), exist_ok=True
    )
    fmri_glm = process_ses(sub, ses, path_to_data)


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
