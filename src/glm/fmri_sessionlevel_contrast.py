import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
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
from src.annotations.annotations import trim_events_df




 # Set constants
sub = 'sub-01'
actions = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
dpath = path_to_data + 'shinobi/'
contrast = 'Jump'
if not os.path.isdir(path_to_data + 'processed/cmaps/' + contrast):
    os.mkdir(path_to_data + 'processed/cmaps/' + contrast)


seslist= os.listdir(dpath + sub)
# load nifti imgs
for ses in ['ses-001', 'ses-002', 'ses-003', 'ses-004']:#sorted(seslist):
    runs = [filename[-13] for filename in os.listdir(dpath + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename]
    fmri_imgs = []
    design_matrices = []
    confounds = []
    confounds_cnames = []
    allruns_events = []
    print('Processing {}'.format(ses))
    print(runs)
    for run in sorted(runs):
        print('run : {}'.format(run))
        data_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = dpath + 'derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = '/project/rrg-pbellec/hyruuk/hyruuk_shinobi_behav/data/anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        fmri_img = image.concat_imgs(data_fname)
        masker = NiftiMasker()
        masker.fit(anat_fname)
        confounds.append(pd.DataFrame.from_records(load_confounds.Params36().load(confounds_fname)))
        fmri_imgs.append(fmri_img)
        conf=load_confounds.Params36()
        conf.load(confounds_fname)
        confounds_cnames.append(conf.columns_)

        # load events
        with open(path_to_data + 'processed/annotations/{}_{}_run-0{}.pkl'.format(sub, ses, run), 'rb') as f:
            run_events = pickle.load(f)
        if 'Left' in contrast or 'Right' in contrast:
            trimmed_df = trim_events_df(run_events, trim_by='LvR')
        elif 'Jump' in contrast or 'Hit' in contrast:
            trimmed_df = trim_events_df(run_events, trim_by='JvH')
        else:
            trimmed_df = trim_events_df(run_events, trim_by='healthloss')
        allruns_events.append(trimmed_df)



    # create design matrices
    for idx, run in enumerate(sorted(runs)):
        t_r = 1.49
        n_slices = confounds[idx].shape[0]
        frame_times = np.arange(n_slices) * t_r

        design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(frame_times,
                                                                               events=allruns_events[idx],
                                                                              drift_model=None,
                                                                              add_regs=confounds[idx],
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
    try:
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
        cmap.to_filename('data/processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses))
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
    except Exception as e: print(e)
