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
    default="HIT",
    type=str,
    help="Condition (contrast) to process",
)
args = parser.parse_args()

t_r = 1.49
hrf_model = "spm"

def process_subject(sub, condition, path_to_data):
    z_maps_dir = op.join(path_to_data, "processed", "z_maps", "ses-level", condition)
    file_list = os.listdir(z_maps_dir)
    for model in ["full", "simple"]:
        if condition in ['HIT+JUMP', 'RIGHT+LEFT+DOWN']:
            model = "intermediate"
        subjectlevel_z_map_fname = op.join(path_to_data, "processed", "z_maps", "subject-level", condition, f"{sub}_{model}model_{condition}.nii.gz")
        os.makedirs(op.join(path_to_data, "processed", "z_maps", "subject-level", condition), exist_ok=True)
        z_maps = []
        ses_list = []
        for file in sorted(file_list):
            if sub in file and model in file:
                print(f"Adding : {file}")
                z_maps.append(op.join(z_maps_dir, file))

                ses_list.append(file.split("_")[1])
        # Compute map
        second_level_input = z_maps
        column_names = [f"{ses}" for ses in ses_list]
        second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])
        #for idx, ses in enumerate(ses_list):
        #    second_design_matrix[column_names[idx]] = [0] * len(second_level_input)
        #    second_design_matrix[column_names[idx]][idx] = 1


        second_level_model = SecondLevelModel(smoothing_fwhm=None)
        second_level_model = second_level_model.fit(second_level_input,
                                                    design_matrix=second_design_matrix)

        contrast_intercept = np.zeros(len(second_level_input)+1)
        contrast_intercept[0] = 1

        z_map = second_level_model.compute_contrast(second_level_contrast=[1], output_type='z_score')
        z_map.to_filename(subjectlevel_z_map_fname)
        # Create report
        report_path = op.join(shinobi_behav.FIG_PATH, "subject-level", condition, "report")
        os.makedirs(report_path, exist_ok=True)
        report_fname = op.join(report_path, f"{sub}_{model}model_{condition}_report.html")
        report = second_level_model.generate_report(contrasts=['intercept'])
        report.save_as_html(report_fname)

    return z_map


def main():
    z_map = process_subject(sub, condition, path_to_data)

if __name__ == "__main__":
    figures_path = shinobi_behav.FIG_PATH #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    sub = args.subject
    condition = args.condition

    # Log job info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {condition}")
    print(f"Writing processed data in : {path_to_data}")
    print(f"Writing reports in : {figures_path}")
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

