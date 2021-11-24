import pandas as pd
import os.path as op
import shinobi_behav
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.input_data import NiftiMasker
import load_confounds
import pickle
import nilearn
from scipy import signal
from scipy.stats import zscore
from nilearn.glm.second_level import SecondLevelModel
import argparse

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
    default='Jump',
    type=str,
    help="Contrast or conditions to compute",
)
args = parser.parse_args()


figures_path = shinobi_behav.figures_path
path_to_data = shinobi_behav.path_to_data
 # Set constants
sub = 'sub-' + args.subject
actions = shinobi_behav.actions
contrast = args.contrast

if not os.path.isdir(path_to_data + 'processed/z_maps/subject-level/' + contrast):
    os.makedirs(path_to_data + 'processed/z_maps/subject-level/' + contrast)

if not os.path.isdir(figures_path + '/subject-level/' + contrast):
    os.makedirs(figures_path + '/subject-level/' + contrast)

z_maps = []
# load nifti imgs
files = os.listdir(path_to_data + 'processed/z_maps/run-level-allregs/' + contrast)
for file in files:#,'ses-006','ses-007','ses-008']:#sorted(seslist):
    if sub in file:
        z_map_name = path_to_data + 'processed/z_maps/run-level-allregs/' + contrast + '/' + file
        z_maps.append(z_map_name)

second_level_input = z_maps
second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['intercept'])

second_level_model = SecondLevelModel(smoothing_fwhm=None)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=second_design_matrix)

z_map_name = path_to_data + '/processed/z_maps/subject-level/{}/{}_{}.nii.gz'.format(contrast, sub, contrast)
z_map = second_level_model.compute_contrast(output_type='z_score')
z_map.to_filename(z_map_name)
print('Saved {}'.format(z_map_name))
report = second_level_model.generate_report(contrasts=['intercept'])
report.save_as_html(figures_path + '/subject-level/{}_{}_slm.html'.format(sub, contrast))

# compute thresholds
clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')

# save images
print('Generating views')
view = plotting.view_img(clean_map, threshold=3, title='{} contrast (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
view.save_as_html(figures_path + '/{}_{}_slm_FDRcluster_fwhm5.html'.format(sub, contrast))
# save also uncorrected map
view = plotting.view_img(uncorr_map, threshold=3, title='{} contrast (p<0.001), uncorr'.format(contrast))
view.save_as_html(figures_path + '/{}_{}_slm_uncorr_fwhm5.html'.format(sub, contrast))
print('Done')
