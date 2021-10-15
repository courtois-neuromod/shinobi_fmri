import os
import pandas as pd
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import trim_events_df
import numpy as np
import pdb
import argparse
import nilearn
import shinobi_behav
from nilearn.glm.first_level import make_first_level_design_matrix

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



# Set constants
figures_path = shinobi_behav.figures_path#'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
path_to_data = shinobi_behav.path_to_data#'/media/storage/neuromod/shinobi_data/'
sub = 'sub-' + args.subject
contrast = args.contrast

if not os.path.isdir(path_to_data + 'processed/cmaps/' + contrast):
    os.makedirs(path_to_data + 'processed/cmaps/' + contrast)

seslist = os.listdir(path_to_data + 'shinobi/' + sub)

for ses in sorted(seslist): #['ses-001', 'ses-002', 'ses-003', 'ses-004']:
    runs = [filename[-12] for filename in os.listdir(path_to_data + '/shinobi/{}/{}/func'.format(sub, ses)) if 'events.tsv' in filename]
    print('Processing {} {}'.format(sub, ses))
    for run in sorted(runs):
        print('Run : {}'.format(run))
        data_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = path_to_data + 'anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        cmap_fname = path_to_data + 'processed/cmaps/{}/{}_{}_run-0{}.nii.gz'.format(contrast, sub, ses, run)
        events_fname = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
        if not os.path.exists(cmap_fname):
            run_events = pd.read_csv(events_fname)
            if not run_events.empty:
                try:
                    fmri_img = image.concat_imgs(data_fname)
                    bold_shape = fmri_img.shape
                    confounds = Confounds(strategy=['high_pass', 'motion', 'global', 'wm_csf'],
                                                    motion="full", wm_csf='basic',
                                                    global_signal='full').load(data_fname)
                    # trim events
                    if 'Left' in contrast or 'Right' in contrast:
                        trimmed_df = trim_events_df(run_events, trim_by='LvR').iloc[: , 1:]
                    elif 'Jump' in contrast or 'Hit' in contrast:
                        trimmed_df = trim_events_df(run_events, trim_by='JvH').iloc[: , 1:]
                    elif 'HealthLoss' in contrast:
                        trimmed_df = trim_events_df(run_events, trim_by='healthloss').iloc[: , 1:]

                    # create design matrix
                    n_slices = bold_shape[-1]
                    t_r = 1.49
                    frame_times = np.arange(n_slices) * t_r
                    design_matrix = make_first_level_design_matrix(frame_times,
                                                                events=trimmed_df,
                                                                drift_model=None,
                                                                add_regs=confounds,
                                                                add_reg_names=None)

                    fmri_glm = FirstLevelModel(t_r=1.49,
                                               noise_model='ar1',
                                               standardize=False,
                                               hrf_model='spm',
                                               drift_model=None,
                                               high_pass=None,
                                               n_jobs=16,
                                               smoothing_fwhm=5,
                                               mask_img=anat_fname)
                    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)

                    cmap = fmri_glm.compute_contrast(contrast,
                                              stat_type='F',
                                              output_type='z_score')
                    cmap.to_filename(cmap_fname)
                    print('cmap saved')
                    report = fmri_glm.generate_report(contrasts=[contrast])
                    report.save_as_html(figures_path + '/{}_{}_{}_flm.html'.format(sub, ses, contrast))

                    # get stats map
                    z_map = fmri_glm.compute_contrast(contrast,
                        output_type='z_score', stat_type='F')

                    # compute thresholds
                    clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
                    uncorr_map, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')

                    # save images
                    print('Generating views')
                    view = plotting.view_img(clean_map, threshold=3, title='{} (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
                    view.save_as_html(figures_path + '/{}_{}_{}_flm_FDRcluster_fwhm5.html'.format(sub, ses, contrast))
                    # save also uncorrected map
                    view = plotting.view_img(uncorr_map, threshold=3, title='{} (p<0.001), uncorr'.format(contrast))
                    view.save_as_html(figures_path + '/{}_{}_{}_flm_uncorr_fwhm5.html'.format(sub, ses, contrast))
                except Exception as e:
                    print(e)
                    print('Session map not computed.')
