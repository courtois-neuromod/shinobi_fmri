import os
import pandas as pd
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import trim_events_df, get_scrub_regressor
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
    default='Hit',
    type=str,
    help="Contrast or conditions to compute",
)
args = parser.parse_args()



# Set constants
figures_path = shinobi_behav.figures_path#'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
path_to_data = shinobi_behav.path_to_data#'/media/storage/neuromod/shinobi_data/'
sub = 'sub-' + args.subject
contrast = args.contrast
t_r = 1.49

if not os.path.isdir(path_to_data + 'processed/z_maps/run-level-allregs/' + contrast):
    os.makedirs(path_to_data + 'processed/z_maps/run-level-allregs/' + contrast)

if not os.path.isdir(figures_path + '/run-level-allregs/' + contrast):
    os.makedirs(figures_path + '/run-level-allregs/' + contrast)

if not os.path.isdir(figures_path + 'design_matrices-allregs'):
    os.makedirs(figures_path + 'design_matrices-allregs')

seslist = os.listdir(path_to_data + 'shinobi/' + sub)

for ses in sorted(seslist): #['ses-001', 'ses-002', 'ses-003', 'ses-004']:
    runs = [filename[-12] for filename in os.listdir(path_to_data + '/shinobi/{}/{}/func'.format(sub, ses)) if 'events.tsv' in filename]
    print('Processing {} {}'.format(sub, ses))
    for run in sorted(runs):
        print('Run : {}'.format(run))
        data_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        confounds_fname = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_desc-confounds_timeseries.tsv'.format(sub, ses, sub, ses, run)
        anat_fname = path_to_data + 'anat/derivatives/fmriprep-20.2lts/fmriprep/{}/anat/{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub)
        z_map_fname = path_to_data + 'processed/z_maps/run-level-allregs/{}/{}_{}_run-0{}.nii.gz'.format(contrast, sub, ses, run)
        events_fname = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
        if os.path.exists(z_map_fname):
            print('z_map already exists')
        else:
            run_events = pd.read_csv(events_fname)
            if run_events.empty:
                print('run_events is empty')
            else:
                try:
                    #confounds = Confounds(strategy=['wm_csf'], # KEEP TRYING THIS
                    confounds = Confounds(strategy=['high_pass', 'motion'],
                                                        motion="full", wm_csf='full',
                                                        global_signal='full').load(data_fname)


                    #fmri_img = image.concat_imgs(data_fname)
                    fmri_img = clean_img(data_fname, standardize=True, detrend=True, high_pass=None, t_r=t_r, ensure_finite=True, confounds=None)

                    #fmri_img = image.resample_img(fmri_img, target_affine=fmri_img.affine, target_shape=(10, 10, 10))
                    #anat_ds = image.resample_img(anat_fname, target_affine=fmri_img.affine, target_shape=(10, 10, 10))
                    #fmri_img = fmri_ds

                    mean_img = image.mean_img(data_fname)
                    bold_shape = fmri_img.shape


                    # forcer confounds en float (ou autres types)
                    hrf_model = 'spm'
                    trimmed_df = trim_events_df(run_events, trim_by='event')
                    # create design matrix
                    n_slices = bold_shape[-1]
                    frame_times = np.arange(n_slices) * t_r
                    design_matrix = make_first_level_design_matrix(frame_times,
                                                                events=trimmed_df,
                                                                drift_model=None,
                                                                hrf_model=hrf_model,
                                                                add_regs=confounds,
                                                                add_reg_names=None)

                    # save design matrix plot
                    dm_fname = figures_path + 'design_matrices-allregs' + '/dm-preclean_plot_{}_{}_run-0{}_{}.png'.format(sub, ses, run, contrast)
                    plotting.plot_design_matrix(design_matrix, output_file=dm_fname)

                    clean_regs = clean(design_matrix.to_numpy(), detrend=True, standardize=True, high_pass=None, t_r=t_r, ensure_finite=True, confounds=None) # remove signal.clean when passing confounds to DM
                    clean_designmat = pd.DataFrame(clean_regs, columns=design_matrix.columns.to_list())
                    clean_designmat['constant'] = 1 # rename design_matrix_clean
                    #design_matrix = clean_designmat
                    #design_matrix = get_scrub_regressor(run_events, design_matrix)

                    dm_fname = figures_path + 'design_matrices-allregs' + '/dm_plot_{}_{}_run-0{}_{}.png'.format(sub, ses, run, contrast)
                    plotting.plot_design_matrix(design_matrix, output_file=dm_fname)


                    # fit glm
                    fmri_glm = FirstLevelModel(t_r=1.49,
                                               noise_model='ar1',
                                               standardize=False,
                                               hrf_model=hrf_model,
                                               drift_model=None,
                                               high_pass=None,
                                               n_jobs=16,
                                               smoothing_fwhm=5,
                                               mask_img=anat_fname,
                                               minimize_memory=False)
                    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)

                    # get stats map
                    z_map = fmri_glm.compute_contrast(contrast, output_type='z_score', stat_type='F')
                    z_map.to_filename(z_map_fname)
                    print('z_map saved')
                    report = fmri_glm.generate_report(contrasts=[contrast])
                    report.save_as_html(figures_path + '/run-level-allregs/{}/{}_{}_run-0{}_{}_flm.html'.format(contrast, sub, ses, run, contrast))



                    # save images
                    print('Generating views')
                    clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)
                    view = plotting.view_img(clean_map, threshold=3, title='{} (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
                    view.save_as_html(figures_path + '/run-level-allregs/{}/{}_{}_run-0{}_{}_flm_FDRcluster_fwhm5.html'.format(contrast, sub, ses, run, contrast))
                    # save also uncorrected map
                    uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')
                    view = plotting.view_img(uncorr_map, threshold=3, title='{} (p<0.001), uncorr'.format(contrast))
                    view.save_as_html(figures_path + '/run-level-allregs/{}/{}_{}_run-0{}_{}_flm_uncorr_fwhm5.html'.format(contrast, sub, ses, run, contrast))


                    ### Extract activation clusters
                    table = get_clusters_table(z_map, stat_threshold=3.1,
                                               cluster_threshold=20).set_index('Cluster ID', drop=True)
                    # get the 3 largest clusters' max x, y, and z coordinates
                    coords = table.loc[range(1, 4), ['X', 'Y', 'Z']].values
                    # extract time series from each coordinate
                    masker = input_data.NiftiSpheresMasker(coords)
                    real_timeseries = masker.fit_transform(fmri_img)
                    predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])
                    # TODO : fit transform de la z_map pour obtenir la valeur du test

                    # Plot figure
                    # colors for each of the clusters
                    colors = ['blue', 'navy', 'purple', 'magenta', 'olive', 'teal']
                    # plot the time series and corresponding locations

                    fig1, axs1 = plt.subplots(2, 3)
                    for i in range(0, 3):
                        # plotting time series
                        axs1[0, i].set_title('Cluster peak {}\n'.format(coords[i]))
                        axs1[0, i].plot(real_timeseries[:, i], c=colors[i], lw=2)
                        axs1[0, i].plot(predicted_timeseries[:, i], c='r', ls='--', lw=2)
                        axs1[0, i].set_xlabel('Time')
                        axs1[0, i].set_ylabel('Signal intensity', labelpad=0)
                        # plotting image below the time series
                        roi_img = plotting.plot_stat_map(
                            z_map, cut_coords=[coords[i][2]], threshold=3.1, figure=fig1,
                            axes=axs1[1, i], display_mode='z', colorbar=False, bg_img=mean_img)
                        roi_img.add_markers([coords[i]], colors[i], 300)

                    fig1.set_size_inches(24, 14)
                    signals_plot_name = figures_path + 'run-level-allregs/{}/signals_{}_{}_run-0{}.png'.format(contrast, sub, ses, run)
                    fig1.savefig(signals_plot_name)
                except Exception as e:
                    print(e)
                    print('Run map not computed.')
