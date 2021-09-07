import pandas as pd
import os.path as op
from src.params import figures_path, path_to_data
from nilearn import plotting
from nilearn import image
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
from nistats.thresholding import map_threshold
from nilearn.glm.first_level import FirstLevelModel
from nilearn.input_data import NiftiMasker
import load_confounds
import pickle
import nilearn
from scipy import signal
from scipy.stats import zscore
from nilearn.glm.second_level import SecondLevelModel


 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = path_to_data + 'shinobi/'
contrast = 'Hit'


#seslist= os.listdir(dpath + sub)
cmaps = []
# load nifti imgs
for ses in ['ses-002','ses-003','ses-004','ses-005','ses-006','ses-007','ses-008']:#sorted(seslist):
    cmap_name = path_to_data + 'processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
    cmaps.append(cmap_name)


second_level_input = cmaps
second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['intercept'])


second_level_model = SecondLevelModel(smoothing_fwhm=None)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=second_design_matrix)

z_map = second_level_model.compute_contrast(output_type='z_score')
z_map.to_filename('data/processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, contrast))
report = second_level_model.generate_report(contrasts=['intercept'])
report.save_as_html(figures_path + '/{}_{}_slm.html'.format(sub, contrast))

# compute thresholds
clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

# save images
print('Generating views')
view = plotting.view_img(clean_map, threshold=3, title='{} contrast (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
view.save_as_html(figures_path + '/{}_{}_slm_FDRcluster_fwhm5.html'.format(sub, contrast))
# save also uncorrected map
view = plotting.view_img(uncorr_map, threshold=3, title='{} contrast (p<0.001), uncorr'.format(contrast))
view.save_as_html(figures_path + '/{}_{}_slm_uncorr_fwhm5.html'.format(sub, contrast))
