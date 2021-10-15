import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
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
from shinobi_fmri.annotations.annotations import trim_events_df
import argparse
import pdb
from tqdm import tqdm
from nilearn.signal import clean
from nilearn.image import clean_img

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

contrast = args.contrast
t_r = 1.49

if not os.path.isdir(path_to_data + 'processed/cmaps_COSINES/' + contrast):
    os.makedirs(path_to_data + 'processed/cmaps_COSINES/' + contrast)


seslist = os.listdir(path_to_data + 'shinobi/' + sub)
# load nifti imgs
for ses in sorted(seslist):
    cmap_fname = path_to_data + 'processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
    if not os.path.exists(cmap_fname):
        runs = [filename[-12] for filename in os.listdir(path_to_data + '/shinobi/{}/{}/func'.format(sub, ses)) if 'events.tsv' in filename]
        fmri_imgs = []
        design_matrices = []
        confounds = []
        confounds_cnames = []
        allruns_events = []
        print('Processing {}'.format(ses))
        print('Runs to process : {}'.format(runs))
        for run in sorted(runs):
            data_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
            confounds_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
            anat_fname = path_to_data + 'anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
            events_fname = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
            # Open annotations
            run_events = pd.read_csv(events_fname)
            if not run_events.empty:
                print('run : {}'.format(run))
                fmri_img = image.concat_imgs(data_fname)

                bold_shape = fmri_img.shape

                fmri_imgs.append(fmri_img)
                # trim events
                if 'Left' in contrast or 'Right' in contrast:
                    trimmed_df = trim_events_df(run_events, trim_by='LvR').iloc[: , 1:]
                elif 'Jump' in contrast or 'Hit' in contrast:
                    trimmed_df = trim_events_df(run_events, trim_by='JvH').iloc[: , 1:]
                elif 'HealthLoss' in contrast:
                    trimmed_df = trim_events_df(run_events, trim_by='healthloss').iloc[: , 1:]
                allruns_events.append(trimmed_df)


                # create design matrices
                n_slices = bold_shape[-1]
                frame_times = np.arange(n_slices) * t_r

                design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(frame_times,
                events=trimmed_df,
                drift_model='cosine',
                add_regs=None,
                add_reg_names=None)
                design_matrices.append(design_matrix)

            else:
                print('Events dataframe empty for {} {} run-0{}.'.format(sub, ses, run))
            #pdb.set_trace()



        try:
            # build model
            print('Fitting a GLM')
            fmri_glm = FirstLevelModel(t_r=1.49,
                                       noise_model='ar1',
                                       standardize=True,
                                       hrf_model='spm',
                                       drift_model=None,
                                       high_pass=.01,
                                       n_jobs=16,
                                       smoothing_fwhm=5,
                                       mask_img=anat_fname)
            fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

            cmap = fmri_glm.compute_contrast(contrast,
                                              stat_type='F',
                                              output_type='z_score')
            cmap.to_filename(cmap_fname)
            print('cmap saved')
            report = fmri_glm.generate_report(contrasts=[contrast])
            report.save_as_html(figures_path + 'cosines' + '/{}_{}_{}_flm.html'.format(sub, ses, contrast))

            # get stats map
            z_map = fmri_glm.compute_contrast(contrast,
                output_type='z_score', stat_type='F')

            # compute thresholds
            clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
            uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')

            # save images
            print('Generating views')
            view = plotting.view_img(clean_map, threshold=3, title='{} (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
            view.save_as_html(figures_path + 'cosines' + '/{}_{}_{}_flm_FDRcluster_fwhm5.html'.format(sub, ses, contrast))
            # save also uncorrected map
            view = plotting.view_img(uncorr_map, threshold=3, title='{} (p<0.001), uncorr'.format(contrast))
            view.save_as_html(figures_path + 'cosines' + '/{}_{}_{}_flm_uncorr_fwhm5.html'.format(sub, ses, contrast))

        except Exception as e:
            print(e)
            print('Session map not computed.')
