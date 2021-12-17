import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import shinobi_behav
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.input_data import NiftiMasker
import pickle
import nilearn
from scipy import signal
from scipy.stats import zscore
from shinobi_fmri.annotations.annotations import trim_events_df, get_scrub_regressor
import argparse
import pdb
from load_confounds import Confounds
from nilearn.image import clean_img
from nilearn.glm import threshold_stats_img
from nilearn.signal import clean
import nibabel as nb
from nilearn.plotting import plot_img_on_surf, plot_stat_map
#import shinobi_fmri

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='01',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-c",
    "--contrast",
    default='Kill',
    type=str,
    help="Contrast or conditions to compute",
)
args = parser.parse_args()

figures_path = shinobi_behav.figures_path#'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'#
path_to_data = shinobi_behav.path_to_data #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/data/'#

 # Set constants
sub = 'sub-' + args.subject
contrast = args.contrast
t_r = 1.49
hrf_model = 'spm'

for ses_id in range(15):
    try:
        ses = f'ses-0{ses_id:02d}'

        z_map_fname = path_to_data + 'processed/z_maps/session-level-allregs/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
        z_map = nb.load(z_map_fname)

        ### Plots
        anat_fname = op.join(
            path_to_data,
            "anat",
            "derivatives",
            "fmriprep-20.2lts",
            "fmriprep",
            sub,
            "anat",
            f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
        )
        bg_img = nb.load(anat_fname)

        # Raw map
        # Plot surface
        plot_img_on_surf(z_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast,f'{sub}_{ses}_{contrast}.png'))
        plot_stat_map(z_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast, f'{sub}_{ses}_{contrast}_slices.png'))


        # Report
        #   report = fmri_glm.generate_report(contrasts=[contrast])
        #report.save_as_html(figures_path + '/session-level-allregs' + '/{}/{}_{}_{}_flm.html'.format(contrast, sub, ses, contrast))

        # compute thresholds
        clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=5)
        uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')

        # save images
        print('Generating views')
        view = plotting.view_img(clean_map, threshold=3, title='{} (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
        view.save_as_html(op.join(figures_path, 'session-level-allregs',contrast, f'{sub}_{ses}_{contrast}_flm_FDRcluster_fwhm5.html'))
        plot_img_on_surf(clean_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast,f'{sub}_{ses}_{contrast}_FDR.png'))
        plot_stat_map(clean_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast, f'{sub}_{ses}_{contrast}_slices_FDR.png'))
        # save also uncorrected map
        view = plotting.view_img(uncorr_map, threshold=3, title='{} (p<0.001), uncorr'.format(contrast))
        view.save_as_html(op.join(figures_path, 'session-level-allregs', contrast, f'{sub}_{ses}_{contrast}_flm_uncorr_fwhm5.html'))
        plot_img_on_surf(uncorr_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast, f'{sub}_{ses}_{contrast}_uncorr.png'))
        plot_stat_map(uncorr_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'session-level-allregs', contrast, f'{sub}_{ses}_{contrast}_slices_uncorr.png'))
    except FileNotFoundError as e:
        print(e)
