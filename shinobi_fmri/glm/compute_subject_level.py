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
import shinobi_behav

# Suppress informational warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
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
    z_maps_dir = op.join(path_to_data, "processed", "z_maps", "ses-level", condition)
    
    if not op.exists(z_maps_dir):
        msg = f"Directory not found: {z_maps_dir}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None
        
    file_list = os.listdir(z_maps_dir)
    z_map = None

    subjectlevel_z_map_fname = op.join(path_to_data, "processed", "z_maps", "subject-level", condition, f"{sub}_{condition}.nii.gz")
    os.makedirs(op.join(path_to_data, "processed", "z_maps", "subject-level", condition), exist_ok=True)

    if logger:
        logger.log_computation_start(f"SubjectLevel_{condition}", subjectlevel_z_map_fname)
    else:
        print(f"Computing subject level for {condition}")

    z_maps = []
    ses_list = []
    for file in sorted(file_list):
        if sub in file:
            if logger:
                logger.info(f"Adding : {file}")
            else:
                print(f"Adding : {file}")
            z_maps.append(op.join(z_maps_dir, file))
            ses_list.append(file.split("_")[1])

    if not z_maps:
        msg = f"No z-maps found for {sub} {condition}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

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

        # Create report
        report_path = op.join(shinobi_behav.FIG_PATH, "subject-level", condition, "report")
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
    figures_path = shinobi_behav.FIG_PATH #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
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
