"""
Session-Level (Second-Level) GLM Analysis for fMRI Data

This script performs second-level GLM analysis by combining multiple runs
within a single session of the Shinobi video game task.

STATISTICAL METHODS:
-------------------
GLM Specification:
  - Model: First-level GLM applied across multiple runs (fixed-effects)
  - TR: 1.49 seconds
  - Smoothing: 5mm FWHM isotropic Gaussian kernel
  - Noise model: AR(1) autoregressive model
  - Analysis level: Within-session, combining all runs

Design Matrix:
  - Task regressors: Game events from all runs concatenated
  - Confounds: Standard fMRIPrep confounds per run
  - Run boundaries: Handled automatically by nilearn

Statistical Inference:
  - Contrast type: F-test for each condition across runs
  - Multiple comparison correction: Cluster-level FWE correction
  - Cluster-forming threshold: Z > 2.3 (liberal for session-level)
  - Family-wise error rate: alpha = 0.05
  - Effect: Fixed-effects combining evidence across runs

Outputs:
  - Beta maps: Session-level effect sizes
  - Z-maps: Uncorrected and corrected statistical maps
  - HTML reports: Interactive visualizations
  - Metadata JSON: Complete provenance tracking
  - Dataset description: BIDS-compliant metadata

USAGE:
------
  python compute_session_level.py -s sub-01 -ses ses-001 -v
  python compute_session_level.py --subject sub-01 --session ses-001 --save-glm

For detailed usage, see TASKS.md or run: python compute_session_level.py --help
"""

import os
import os.path as op
from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import warnings
import re

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_sidecar_metadata, create_dataset_description
from shinobi_fmri.glm import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress informational warnings
warnings.filterwarnings('ignore', message='.*imgs are being resampled to the mask_img resolution.*')
warnings.filterwarnings('ignore', message='.*Mean values of 0 observed.*')
warnings.filterwarnings('ignore', message='.*design matrices are supplied.*')
warnings.filterwarnings("ignore", category=FutureWarning)

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


def get_output_names(
    sub: str,
    ses: str,
    regressor_output_name: str,
    n_runs: Optional[int] = None,
    use_low_level_confs: bool = False
) -> Tuple[str, str, str, str]:
    """
    Construct BIDS-compliant output file paths for session-level GLM results.

    Args:
        sub: Subject identifier (e.g., 'sub-01')
        ses: Session identifier (e.g., 'ses-001')
        regressor_output_name: Name of the contrast/regressor
        n_runs: Number of runs included (None for all runs in session)
        use_low_level_confs: Whether low-level confounds were used

    Returns:
        Tuple of (glm_fname, z_map_fname, beta_map_fname, report_fname):
            Paths to GLM pickle, z-map, beta-map, and HTML report files

    Note:
        Output directory structure follows BIDS derivatives convention:
        {DATA_PATH}/processed/session-level/sub-XX/ses-YY/{z_maps,beta_maps,glm}/
    """
    # Determine output directory based on whether low-level confounds are used
    output_dir = "processed_low-level" if use_low_level_confs else "processed"

    # Determine level directory name
    if n_runs is None:
        level_dir = "session-level"
    else:
        level_dir = f"session-level_{n_runs}runs"

    # Directory structure: processed/session-level/sub-XX/ses-YY/z_maps/ and beta_maps/
    z_maps_dir = op.join(config.DATA_PATH, output_dir, level_dir, sub, ses, "z_maps")
    beta_maps_dir = op.join(config.DATA_PATH, output_dir, level_dir, sub, ses, "beta_maps")
    glm_dir = op.join(config.DATA_PATH, output_dir, level_dir, sub, ses, "glm")
    os.makedirs(z_maps_dir, exist_ok=True)
    os.makedirs(beta_maps_dir, exist_ok=True)
    os.makedirs(glm_dir, exist_ok=True)

    # Optional descriptor for incremental analysis (number of runs)
    desc_suffix = f"_desc-{n_runs}runs" if n_runs is not None else ""

    # BIDS-compliant base filename
    base_name = f"{sub}_{ses}_task-shinobi{desc_suffix}_contrast-{regressor_output_name}"

    glm_fname = op.join(glm_dir, f"{base_name}_glm.pkl")
    z_map_fname = op.join(z_maps_dir, f"{base_name}_stat-z.nii.gz")
    beta_map_fname = op.join(beta_maps_dir, f"{base_name}_stat-beta.nii.gz")

    # Report filename (keeping reports in figures_path)
    report_dir = op.join(config.FIG_PATH, level_dir, regressor_output_name, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_fname = op.join(report_dir, f"{base_name}_report.html")

    return glm_fname, z_map_fname, beta_map_fname, report_fname


def load_session(sub, ses, run_list, path_to_data, use_low_level_confs=False):
    """
    Loads and prepares the data for a given session.
    """
    design_matrices = []
    fmri_imgs = []
    anat_fname = None
    
    for run in run_list:
        fmri_fname, anat_fname, events_fname, mask_fname = utils.get_filenames(
            sub, ses, run, path_to_data
        )
        
        load_run_result = utils.load_run(
            fmri_fname, mask_fname, events_fname, path_to_data, CONDS_LIST, use_low_level_confs=use_low_level_confs
        )
        design_matrix_clean, fmri_img, mask_resampled = load_run_result
        design_matrices.append(design_matrix_clean)
        fmri_imgs.append(fmri_img)
    return fmri_imgs, design_matrices, mask_resampled, anat_fname


def remove_runs_without_target_regressor(
    regressor_names, fmri_imgs, trimmed_design_matrices
):
    """
    Removes runs that do not contain all target regressors.

    Args:
        regressor_names: List of regressor names that must be present
        fmri_imgs: List of fMRI image objects
        trimmed_design_matrices: List of design matrices (pandas DataFrames)

    Returns:
        Tuple of (filtered_images, filtered_design_matrices) containing only
        runs that have all required regressors.

    Note:
        Uses index-based filtering to avoid pandas DataFrame comparison issues
        when checking membership with 'in' operator.
    """
    # Track indices of runs to keep
    indices_to_keep = []

    for i, (img, df) in enumerate(zip(fmri_imgs, trimmed_design_matrices)):
        # Check if this run has all required regressors
        has_all_regressors = all(reg in df.columns for reg in regressor_names)
        if has_all_regressors:
            indices_to_keep.append(i)

    # Keep only runs that have all target regressors
    images_filtered = [fmri_imgs[i] for i in indices_to_keep]
    dataframes_filtered = [trimmed_design_matrices[i] for i in indices_to_keep]

    return images_filtered, dataframes_filtered


def trim_design_matrices(
    design_matrices: List[pd.DataFrame],
    regressor_name: str,
    conditions_list: List[str]
) -> List[pd.DataFrame]:
    """
    Remove unwanted condition regressors from design matrices.

    For single-condition GLM analysis, this removes all other condition regressors
    while keeping the target condition and all confounds. This prevents
    multi-collinearity and focuses the GLM on the contrast of interest.

    Args:
        design_matrices: List of design matrices (one per run)
        regressor_name: Name of the regressor to keep (e.g., 'HIT')
        conditions_list: List of all conditions in the analysis

    Returns:
        List of trimmed design matrices with only target regressor + confounds

    Note:
        Confound regressors (motion, WM, CSF, etc.) are always kept.
        Other condition regressors are removed to create orthogonal contrasts.
    """
    regressors_to_remove = conditions_list.copy()
    if not "lvl" in regressor_name:
        if regressor_name in regressors_to_remove:
            regressors_to_remove.remove(regressor_name)

    trimmed_design_matrices = []
    for design_matrix in design_matrices:
        trimmed_design_matrix = design_matrix
        for reg in regressors_to_remove:
            try:
                trimmed_design_matrix = trimmed_design_matrix.drop(columns=reg)
            except Exception as e:
                pass
        trimmed_design_matrices.append(trimmed_design_matrix)
    return trimmed_design_matrices


def make_or_load_glm(
    sub: str,
    ses: str,
    run_list: List[str],
    glm_regressors: List[str],
    glm_fname: str,
    mask_resampled_global: Optional[Any] = None,
    save_glm: bool = False,
    use_low_level_confs: bool = False,
    conditions_list: Optional[List[str]] = None
) -> Tuple[Any, Any]:
    """
    Create or load a fitted GLM for session-level analysis.

    If save_glm is enabled and a saved GLM exists, loads it from disk.
    Otherwise, loads data, fits a new GLM, and optionally saves it.

    Args:
        sub: Subject identifier (e.g., 'sub-01')
        ses: Session identifier (e.g., 'ses-001')
        run_list: List of run numbers to include
        glm_regressors: List of regressor names for this GLM
        glm_fname: Path where GLM pickle file is/will be saved
        mask_resampled_global: Pre-computed mask (if available)
        save_glm: Whether to save/load GLM objects (default: False)
        use_low_level_confs: Whether to include psychophysical confounds
        conditions_list: List of all conditions in the analysis (for trimming)

    Returns:
        Tuple of (fitted_glm, mask_resampled)

    Note:
        For interaction contrasts (e.g., 'HITXlvl5'), creates an interaction
        regressor by multiplying the two base regressors.
    """
    if save_glm and os.path.exists(glm_fname):
        with open(glm_fname, "rb") as f:
            fmri_glm = pickle.load(f)
        mask_resampled = mask_resampled_global
    else:
        # Compute GLM fresh
        fmri_imgs, design_matrices, mask_resampled, anat_fname = load_session(
            sub, ses, run_list, path_to_data, use_low_level_confs=use_low_level_confs
        )

        # Use provided conditions_list or default to game conditions
        if conditions_list is None:
            conditions_list = config.CONDITIONS

        trimmed_design_matrices = trim_design_matrices(
            design_matrices, glm_regressors[0], conditions_list
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

        mask_to_use = mask_resampled_global if mask_resampled_global is not None else mask_resampled

        fmri_glm = utils.make_and_fit_glm(fmri_imgs, trimmed_design_matrices, mask_to_use)

        if save_glm:
            os.makedirs(os.path.dirname(glm_fname), exist_ok=True)
            with open(glm_fname, "wb") as f:
                pickle.dump(fmri_glm, f, protocol=4)

    return fmri_glm, mask_resampled


def process_ses(sub, ses, path_to_data, save_glm=False, use_low_level_confs=False, logger=None, conditions_list=None, levels_list=None):
    """
    Process an fMRI session for a given subject and session.

    Args:
        conditions_list: List of conditions to process (game conditions or low-level features)
        levels_list: List of levels for interaction terms (optional)
    """
    # Use provided conditions_list or default to game conditions
    if conditions_list is None:
        conditions_list = config.CONDITIONS

    # Use provided levels_list or default to empty (unless specified in config)
    if levels_list is None:
        levels_list = []

    def process_regressor(regressor_name, run_list_subset, n_runs_label, lvl=None):
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

                fmri_glm, _ = make_or_load_glm(sub, ses, run_list_subset, glm_regressors, glm_fname, save_glm=save_glm, use_low_level_confs=use_low_level_confs, conditions_list=conditions_list)

                # Use utils.make_z_map with cluster correction (Liberal threshold for session-level)
                utils.make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_output_name, cluster_thresh=config.GLM_CLUSTER_THRESH_SESSION, alpha=config.GLM_ALPHA)

                # Save metadata JSON sidecar for reproducibility
                metadata = create_metadata(
                    description=f"Session-level GLM z-map for contrast {regressor_output_name}",
                    script_path=__file__,
                    output_files=[z_map_fname, beta_map_fname],
                    parameters={
                        'contrast': regressor_output_name,
                        'glm_regressors': glm_regressors,
                        'cluster_threshold': config.GLM_CLUSTER_THRESH_SESSION,
                        'alpha': config.GLM_ALPHA,
                        'hrf_model': utils.HRF_MODEL,
                        'tr': utils.TR,
                        'use_low_level_confounds': use_low_level_confs,
                        'n_runs': len(run_list_subset),
                    },
                    subject=sub,
                    session=ses,
                    additional_info={
                        'analysis_level': 'session',
                        'runs_included': run_list_subset,
                        'n_runs_label': n_runs_label,
                        'glm_saved': save_glm,
                    }
                )
                save_sidecar_metadata(z_map_fname, metadata, logger=logger)

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
    ses_fpath = op.join(path_to_data, "shinobi.fmriprep", sub, ses, "func")
    ses_files = os.listdir(ses_fpath)
    run_files = [
        x
        for x in ses_files
        if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in x
    ]
    # Extract run numbers
    run_list = []
    for fname in run_files:
        match = re.search(r'run-(\d+)', fname)
        if match:
            run_list.append(match.group(1))
    run_list = sorted(run_list)

    if logger:
        logger.log_run_list(run_list)
    else:
        print(f"Found {len(run_list)} runs for {sub} {ses}: {run_list}")

    # Process incrementally
    for n_runs in range(1, len(run_list) + 1):
        run_list_subset = run_list[:n_runs]
        n_runs_label = None if n_runs == len(run_list) else n_runs

        if not logger:
            print(f"\n{'='*60}")
            print(f"Processing {sub} {ses} with {n_runs} run(s): {run_list_subset}")
            print(f"{'='*60}\n")

        for regressor_name in conditions_list + levels_list:
            process_regressor(regressor_name, run_list_subset, n_runs_label)

        for lvl in levels_list:
            for regressor_name in conditions_list:
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
    logger = AnalysisLogger(
        log_name="GLM_session",
        subject=sub,
        session=ses,
        log_dir=args.log_dir,
        verbosity=log_level
    )

    # Create dataset_description.json for processed outputs
    output_dir = "processed_low-level" if args.low_level_confs else "processed"
    dataset_desc_dir = op.join(path_to_data, output_dir, "session-level")
    if not op.exists(op.join(dataset_desc_dir, "dataset_description.json")):
        create_dataset_description(
            name="Session-level GLM Analysis",
            description="Second-level GLM analysis combining multiple runs within a session (fixed-effects)",
            pipeline_version="0.1.0",
            derived_from="Preprocessed fMRI data (shinobi.fmriprep)",
            parameters={
                'cluster_threshold': config.GLM_CLUSTER_THRESH_SESSION,
                'alpha': config.GLM_ALPHA,
                'hrf_model': utils.HRF_MODEL,
                'tr': utils.TR,
                'use_low_level_confounds': args.low_level_confs,
            },
            output_dir=dataset_desc_dir
        )
        logger.info(f"Created dataset_description.json in {dataset_desc_dir}")

    try:
        process_ses(sub, ses, path_to_data, save_glm=args.save_glm, use_low_level_confs=args.low_level_confs, logger=logger, conditions_list=CONDS_LIST, levels_list=LEVELS)
    finally:
        logger.close()


if __name__ == "__main__":
    figures_path = config.FIG_PATH
    path_to_data = config.DATA_PATH

    # Use low-level conditions when flag is set, otherwise use game conditions
    if args.low_level_confs:
        CONDS_LIST = config.CONDITIONS + config.LOW_LEVEL_CONDITIONS
        LEVELS = []  # No level interactions for low-level features
        additional_contrasts = []
        print("Using GAME AND LOW-LEVEL CONDITIONS as task regressors:")
        print(f"  {CONDS_LIST}")
    else:
        CONDS_LIST = config.CONDITIONS
        LEVELS = []
        additional_contrasts = ["HIT+JUMP", "RIGHT+LEFT+DOWN"]
        print("Using GAME CONDITIONS as task regressors:")
        print(f"  {CONDS_LIST}")

    sub = args.subject
    ses = args.session

    print(f"Processing : {sub} {ses}")
    print(f"Writing processed data in : {path_to_data}")
    print(f"Writing reports in : {figures_path}")

    main()
