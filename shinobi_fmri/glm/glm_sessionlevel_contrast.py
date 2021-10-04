import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import shinobi_behav
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
from shinobi_fmri.annotations.annotations import trim_events_df
from shinobi_behav.params import path_to_data
import argparse
#import shinobi_fmri

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)

args = parser.parse_args()

figures_path = '/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'#shinobi_behav.figures_path
path_to_data = '/media/storage/neuromod/shinobi_data/'#shinobi_behav.path_to_data

 # Set constants
sub = 'sub-' + args.subject

actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = path_to_data + 'shinobi/'
contrast = 'Jump'

if not os.path.isdir(path_to_data + 'processed/cmaps/' + contrast):
    os.makedirs(path_to_data + 'processed/cmaps/' + contrast)


seslist= os.listdir(dpath + sub)
# load nifti imgs
for ses in sorted(seslist): #['ses-001', 'ses-002', 'ses-003', 'ses-004']:
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    fmri_imgs = []
    design_matrices = []
    confounds = []
    confounds_cnames = []
    allruns_events = []
    print('Processing {}'.format(ses))
    print('Runs to process : {}'.format(runs))
    for run in sorted(runs):
        data_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = '/project/rrg-pbellec/hyruuk/hyruuk_shinobi_behav/data/anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        cmap_fname = path_to_data + 'processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
        events_fname = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
        if not os.path.exists(cmap_fname):
            # Open annotations
            run_events = pd.read_csv(events_fname)
            if not run_events.empty:
                print('run : {}'.format(run))
                fmri_img = image.concat_imgs(data_fname)
                masker = NiftiMasker()
                masker.fit(anat_fname)
                confounds.append(pd.DataFrame.from_records(load_confounds.Params36().load(confounds_fname)))
                fmri_imgs.append(fmri_img)
                conf=load_confounds.Params36()
                conf.load(confounds_fname)
                confounds_cnames.append(conf.columns_)

                # load events

                if 'Left' in contrast or 'Right' in contrast:
                    trimmed_df = trim_events_df(run_events, trim_by='LvR')
                elif 'Jump' in contrast or 'Hit' in contrast:
                    trimmed_df = trim_events_df(run_events, trim_by='JvH')
                else:
                    trimmed_df = trim_events_df(run_events, trim_by='healthloss')
                allruns_events.append(trimmed_df)
            else:
                print('Events dataframe empty for {} {} run-0{}.'.format(sub, ses, run))



    # create design matrices
    try:
        for idx, run in enumerate(sorted(runs)):
            t_r = 1.49
            n_slices = confounds[idx].shape[0]
            frame_times = np.arange(n_slices) * t_r

            design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(frame_times,
                                                                                   events=allruns_events[idx],
                                                                                  drift_model=None,
                                                                                  add_regs=confounds[# build modelidx],
                                                                              add_reg_names=confounds_cnames[idx])
            b, a = signal.butter(3, 0.01, btype='high')

            if 'Left' in contrast or 'Right' in contrast:
                LeftH_ts = np.asarray(design_matrix['LeftH'])
                RightH_ts = np.asarray(design_matrix['RightH'])
                LeftH_ts_hpf = signal.filtfilt(b, a, LeftH_ts)
                RightH_ts_hpf = signal.filtfilt(b, a, RightH_ts)
                LeftH_ts_hpf_z = zscore(LeftH_ts_hpf)
                RightH_ts_hpf_z = zscore(RightH_ts_hpf)
                design_matrix['LeftH'] = LeftH_ts_hpf_z
                design_matrix['RightH'] = RightH_ts_hpf_z

            if 'Jump' in contrast or 'Hit' in contrast:
                design_matrix['Jump'] = zscore(signal.filtfilt(b, a, np.asarray(design_matrix['Jump'])))
                design_matrix['Hit'] = zscore(signal.filtfilt(b, a, np.asarray(design_matrix['Hit'])))


            design_matrices.append(design_matrix)



    # build model

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
