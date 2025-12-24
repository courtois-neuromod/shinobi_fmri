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
import shinobi_fmri.config as config

# Suppress informational warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
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
from nilearn.glm.second_level import SecondLevelModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-06",
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-cond",
    "--condition",
    default="DOWNXlvl5",
    type=str,
    help="Condition (contrast) to process",
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
args = parser.parse_args()

t_r = 1.49
hrf_model = "spm"

def process_subject(sub, condition, path_to_data, logger=None):
    # Read session-level z-maps from the new structure
    # We look for session-level (all runs) z-maps for this subject
    session_level_dir = op.join(path_to_data, "processed", "session-level", sub)

    if not op.exists(session_level_dir):
        msg = f"Directory not found: {session_level_dir}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    # Create output directories
    z_maps_out_dir = op.join(path_to_data, "processed", "subject-level", sub, "z_maps")
    beta_maps_out_dir = op.join(path_to_data, "processed", "subject-level", sub, "beta_maps")
    os.makedirs(z_maps_out_dir, exist_ok=True)
    os.makedirs(beta_maps_out_dir, exist_ok=True)

    # Collect all z-maps for this subject and condition
    z_map = None
    z_maps = []
    ses_list = []

    # Iterate through all sessions for this subject
    for ses_dir in sorted(os.listdir(session_level_dir)):
        ses_path = op.join(session_level_dir, ses_dir)
        if not op.isdir(ses_path):
            continue

        # Look for z-maps in the z_maps subdirectory
        z_maps_dir = op.join(ses_path, "z_maps")
        if not op.exists(z_maps_dir):
            continue

        # Find z-map files for this condition
        for file in os.listdir(z_maps_dir):
            if f"contrast-{condition}" in file and file.endswith("stat-z.nii.gz"):
                if logger:
                    logger.info(f"Adding : {file}")
                else:
                    print(f"Adding : {file}")
                z_maps.append(op.join(z_maps_dir, file))
                ses_list.append(ses_dir)

    if not z_maps:
        msg = f"No z-maps found for {sub} {condition}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

    # Output filename
    subjectlevel_z_map_fname = op.join(z_maps_out_dir, f"{sub}_task-shinobi_contrast-{condition}_stat-z.nii.gz")
    subjectlevel_beta_map_fname = op.join(beta_maps_out_dir, f"{sub}_task-shinobi_contrast-{condition}_stat-beta.nii.gz")

    if logger:
        logger.log_computation_start(f"SubjectLevel_{condition}", subjectlevel_z_map_fname)
    else:
        print(f"Computing subject level for {condition}")

    # Compute map
    try:
        second_level_input = z_maps
        column_names = [f"{ses}" for ses in ses_list]
        second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])

        second_level_model = SecondLevelModel(smoothing_fwhm=None)
        second_level_model = second_level_model.fit(second_level_input,
                                                    design_matrix=second_design_matrix)

        z_map = second_level_model.compute_contrast(second_level_contrast=[1],
                                                    output_type='z_score',
                                                    second_level_stat_type="F")
        z_map.to_filename(subjectlevel_z_map_fname)

        if logger:
            logger.log_computation_success(f"SubjectLevel_{condition}", subjectlevel_z_map_fname)

        # Compute and save cluster-corrected Z-map (Conservative threshold)
        try:
            corrected_map = cluster_level_inference(z_map, threshold=3.1, alpha=0.05)
            # BIDS-compliant naming: insert 'desc-corrected'
            corrected_fname = subjectlevel_z_map_fname.replace('_stat-z.nii.gz', '_desc-corrected_stat-z.nii.gz')
            corrected_map.to_filename(corrected_fname)
        except Exception as e:
            print(f"Warning: Failed to compute cluster correction for {condition}: {e}")

        # Compute beta map (using effect size instead of z-score)
        beta_map = second_level_model.compute_contrast(second_level_contrast=[1],
                                                       output_type='effect_size')
        beta_map.to_filename(subjectlevel_beta_map_fname)

        # Create report
        report_path = op.join(config.FIG_PATH, "subject-level", condition, "report")
        os.makedirs(report_path, exist_ok=True)
        report_fname = op.join(report_path, f"{sub}_{condition}_report.html")
        report = second_level_model.generate_report(
            contrasts=['intercept'],
            height_control=None
        )
        report.save_as_html(report_fname)

    except Exception as e:
        if logger:
            logger.log_computation_error(f"SubjectLevel_{condition}", e)
        else:
            print(f"Error: {e}")

    return z_map


def main(logger=None):
    z_map = process_subject(sub, condition, path_to_data, logger=logger)

if __name__ == "__main__":
    figures_path = config.FIG_PATH #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = config.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    sub = args.subject
    condition = args.condition

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="GLM_subject",
        subject=sub,
        condition=condition,
        log_dir=args.log_dir,
        verbosity=log_level
    )
    
    logger.info(f"Processing : {sub} {condition}")
    logger.info(f"Writing processed data in : {path_to_data}")
    logger.info(f"Writing reports in : {figures_path}")
    
    try:
        main(logger=logger)
    finally:
        logger.close()
