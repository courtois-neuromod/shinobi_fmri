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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-01",
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

def main():
    fmri_glm = process_subject(sub, path_to_data)

if __name__ == "__main__":
    figures_path = shinobi_behav.FIG_PATH #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    sub = args.subject
    condition = args.condition

    # Log job info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {ses}")
    print(f"Writing processed data in : {path_to_data}")
    print(f"Writing reports in : {figures_path}")
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

