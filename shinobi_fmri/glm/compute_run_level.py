import os
import os.path as op
import pandas as pd
import warnings
import numpy as np
import argparse
import logging
import pickle
import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import ShinobiLogger
from shinobi_fmri.glm import utils

# Suppress informational warnings
warnings.filterwarnings('ignore', message='.*imgs are being resampled to the mask_img resolution.*')
warnings.filterwarnings('ignore', message='.*Mean values of 0 observed.*')
warnings.filterwarnings('ignore', message='.*design matrices are supplied.*')
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

def process_run(sub, ses, run, path_to_data, save_glm=False, use_low_level_confs=False, logger=None):
    # Determine output directory based on whether low-level confounds are used
    output_dir = "processed_low-level" if use_low_level_confs else "processed"

    # Format run number to be BIDS-compliant (e.g., "2" -> "run-02")
    run_formatted = f"run-{int(run):02d}"

    # Compute GLM for each condition separately
    for regressor_name in CONDS_LIST:
        try:
            # Directory structure: processed/run-level/sub-XX/ses-YY/z_maps/ and beta_maps/
            z_maps_dir = op.join(path_to_data, output_dir, "run-level", sub, ses, "z_maps")
            beta_maps_dir = op.join(path_to_data, output_dir, "run-level", sub, ses, "beta_maps")
            os.makedirs(z_maps_dir, exist_ok=True)
            os.makedirs(beta_maps_dir, exist_ok=True)

            # BIDS-compliant filename
            base_name = f"{sub}_{ses}_task-shinobi_{run_formatted}_contrast-{regressor_name}"
            z_map_fname = op.join(z_maps_dir, f"{base_name}_stat-z.nii.gz")

            if os.path.exists(z_map_fname):
                if logger:
                    logger.log_computation_skip(regressor_name, z_map_fname)
                continue

            # GLM file (if saving)
            glm_dir = op.join(path_to_data, output_dir, "run-level", sub, ses, "glm")
            os.makedirs(glm_dir, exist_ok=True)
            glm_fname = op.join(glm_dir, f"{base_name}_glm.pkl")
            
            fmri_glm = None

            if save_glm and os.path.exists(glm_fname):
                # Load existing GLM
                with open(glm_fname, "rb") as f:
                    if logger:
                        logger.info(f"GLM found, loading : {glm_fname}")
                    fmri_glm = pickle.load(f)
                    if logger:
                        logger.info("Loaded.")
            else:
                # Compute GLM fresh
                fmri_fname, anat_fname, events_fname, mask_fname = utils.get_filenames(
                    sub, ses, run, path_to_data
                )
                if logger:
                    logger.info(f"Loading : {fmri_fname}")
                
                design_matrix_clean, fmri_img, mask_resampled = utils.load_run(
                    fmri_fname, mask_fname, events_fname, path_to_data, CONDS_LIST, use_low_level_confs=use_low_level_confs
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
                            trimmed_design_matrix = trimmed_design_matrix.drop(columns=reg)
                        except Exception as e:
                            if logger:
                                logger.warning(f"{e}\nRegressor {reg} might be missing ?")
                    trimmed_design_matrices.append(trimmed_design_matrix)

                fmri_glm = utils.make_and_fit_glm(
                    fmri_imgs, trimmed_design_matrices, mask_resampled
                )
                if save_glm:
                    with open(glm_fname, "wb") as f:
                        pickle.dump(fmri_glm, f, protocol=4)

            # Compute contrast and z-map
            if logger:
                logger.log_computation_start(regressor_name, z_map_fname)

            # Report filename
            report_dir = op.join(figures_path, "run-level", regressor_name, "report")
            os.makedirs(report_dir, exist_ok=True)
            report_fname = op.join(report_dir, f"{base_name}_report.html")

            # Beta map filename
            beta_map_fname = op.join(beta_maps_dir, f"{base_name}_stat-beta.nii.gz")

            # Use utils.make_z_map with cluster correction (Liberal threshold for run-level)
            z_map = utils.make_z_map(z_map_fname, beta_map_fname, report_fname, fmri_glm, regressor_name, cluster_thresh=2.3, alpha=0.05)
            
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
    # Extract run numbers
    run_list = []
    for fname in run_files:
        match = re.search(r'run-(\d+)', fname)
        if match:
            run_list.append(match.group(1))

    for run in sorted(run_list):
        if logger:
            logger.info(f"Run : {run}")
        else:
            print(f"Run : {run}")
        process_run(sub, ses, run, path_to_data, save_glm=save_glm, use_low_level_confs=use_low_level_confs, logger=logger)
    return

def main(logger=None):
    process_ses(sub, ses, path_to_data, save_glm=args.save_glm, use_low_level_confs=args.low_level_confs, logger=logger)


if __name__ == "__main__":
    figures_path = config.FIG_PATH
    path_to_data = config.DATA_PATH
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
    sub = args.subject
    ses = args.session
    
    print(f"Processing : {sub} {ses}")

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