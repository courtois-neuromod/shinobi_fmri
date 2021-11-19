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
path_to_data = shinobi_behav.path_to_data #'/media/storage/neuromod/shinobi_data/'#

 # Set constants
sub = 'sub-' + args.subject
contrast = args.contrast
t_r = 1.49

if not os.path.isdir(path_to_data + 'processed/z_maps/session-level-allregs/' + contrast):
    os.makedirs(path_to_data + 'processed/z_maps/session-level-allregs/' + contrast)

if not os.path.isdir(figures_path + '/session-level-allregs/' + contrast):
    os.makedirs(figures_path + '/session-level-allregs/' + contrast)


seslist= os.listdir(path_to_data + 'shinobi/' + sub)
# load nifti imgs
for ses in sorted(seslist): #['ses-001', 'ses-002', 'ses-003', 'ses-004']:
    runs = [filename[-12] for filename in os.listdir(path_to_data + '/shinobi/{}/{}/func'.format(sub, ses)) if 'events.tsv' in filename]
    fmri_imgs = []
    design_matrices = []
    confounds = []

    allruns_events = []
    print('Processing {}'.format(ses))
    print('Runs to process : {}'.format(sorted(runs)))
    z_map_fname = path_to_data + 'processed/z_maps/session-level-allregs/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
    for run in sorted(runs):
        data_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = path_to_data + 'anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        events_fname = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
        if not os.path.exists(z_map_fname):
            # Open annotations
            run_events = pd.read_csv(events_fname)
            if not run_events.empty:
                print('Run : {}'.format(run))
                raw_fmri_img = image.concat_imgs(data_fname)
                confound = Confounds(strategy=['high_pass', 'motion'],
                                                motion="full", wm_csf='basic',
                                                global_signal='full').load(data_fname)

                fmri_img = clean_img(clean_img(raw_fmri_img, detrend=False, high_pass=0.01, t_r=t_r, ensure_finite=True, confounds=confound))
                bold_shape = fmri_img.shape
                fmri_imgs.append(fmri_img)

                trimmed_df = trim_events_df(run_events, trim_by='event')
                allruns_events.append(trimmed_df)
                n_slices = confound.shape[0]
                frame_times = np.arange(n_slices) * t_r

                design_matrix = make_first_level_design_matrix(frame_times,
                                                            events=trimmed_df,
                                                            drift_model=None,
                                                            hrf_model=hrf_model,
                                                            add_regs=None,
                                                            add_reg_names=None)

                # save design matrix plot
                clean_regs = clean(design_matrix.to_numpy(), detrend=False, high_pass=0.01, t_r=t_r, ensure_finite=True, confounds=confound)
                clean_designmat = pd.DataFrame(clean_regs, columns=design_matrix.columns.to_list())
                clean_designmat['constant'] = 1
                design_matrix = clean_designmat
                design_matrix = get_scrub_regressor(run_events, design_matrix)
                design_matrices.append(design_matrix)
            else:
                print('Events dataframe empty for {} {} run-0{}.'.format(sub, ses, run))
    print(len(fmri_imgs))
    try:
        #build model
        print('Fitting a GLM')
        fmri_glm = FirstLevelModel(t_r=1.49,
                                   noise_model='ar1',
                                   standardize=False,
                                   hrf_model='spm',
                                   drift_model=None,
                                   high_pass=.01,
                                   n_jobs=16,
                                   smoothing_fwhm=5,
                                   mask_img=anat_fname)
        fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

        z_map = fmri_glm.compute_contrast(contrast,
                                          stat_type='F',
                                          output_type='z_score')
        z_map.to_filename(z_map_fname)
        print('z_map saved')
        report = fmri_glm.generate_report(contrasts=[contrast])
        report.save_as_html(figures_path + '/session-level-allregs' + '/{}_{}_{}_flm.html'.format(sub, ses, contrast))

        # compute thresholds
        clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
        uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')

        # save images
        print('Generating views')
        view = plotting.view_img(clean_map, threshold=3, title='{} (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
        view.save_as_html(figures_path + '/session-level-allregs/' + '/{}/{}_{}_{}_flm_FDRcluster_fwhm5.html'.format(sub, ses, contrast))
        # save also uncorrected map
        view = plotting.view_img(uncorr_map, threshold=3, title='{} (p<0.001), uncorr'.format(contrast))
        view.save_as_html(figures_path + '/session-level-allregs/' + '/{}/{}_{}_{}_flm_uncorr_fwhm5.html'.format(sub, ses, contrast))

    except Exception as e:
        print(e)
        print('Session map not computed.')
