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
from nilearn.plotting import plot_img_on_surf, plot_stat_map
import nibabel as nb

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
parser.add_argument(
    "-f",
    "--from_level",
    default='session',
    type=str,
    help="First-level models to take as input. Can be run or session",
)
args = parser.parse_args()


figures_path = shinobi_behav.figures_path
path_to_data = shinobi_behav.path_to_data
 # Set constants
sub = 'sub-' + args.subject
actions = shinobi_behav.actions
contrast = args.contrast
from_level = args.from_level


os.makedirs(path_to_data + f'processed/z_maps/subject-level-from-{from_level}/' + contrast, exist_ok=True)

os.makedirs(figures_path + f'/subject-level-from-{from_level}/' + contrast, exist_ok=True)
subjects = shinobi_behav.subjects

for sub in subjects:
    for contrast in ['Kill', 'Hit', 'Jump']:
        z_maps = []
        # load nifti imgs
        files = os.listdir(path_to_data + 'processed/z_maps/' + from_level + '-level-allregs/' + contrast)
        for file in files:#,'ses-006','ses-007','ses-008']:#sorted(seslist):
            if sub in file:
                z_map_name = path_to_data + 'processed/z_maps/' + from_level + '-level-allregs/' + contrast + '/' + file
                z_maps.append(z_map_name)

        second_level_input = z_maps
        second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])

        z_map_name = op.join(path_to_data, 'processed', 'z_maps', f'subject-level-from-{from_level}', contrast, f'{sub}_{contrast}.nii.gz')
        if not op.exists(z_map_name):
            second_level_model = SecondLevelModel(smoothing_fwhm=None)
            second_level_model = second_level_model.fit(second_level_input,
                                                        design_matrix=second_design_matrix)
            z_map = second_level_model.compute_contrast(output_type='z_score')
            z_map.to_filename(z_map_name)
            #print(f'Saved {z_map_name}')￼
        else:
            z_map = nb.load(z_map_name)


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


        # Plot surface
        plot_img_on_surf(z_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast,f'{sub}_{contrast}.png'))
        plot_stat_map(z_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast, f'{sub}_{contrast}_slices.png'))

        # Plot report
        report = second_level_model.generate_report(contrasts=['intercept'])
        report.save_as_html(figures_path + f'/subject-level-from-{from_level}/{contrast}/{sub}_{contrast}_slm.html')

        # compute thresholds
        clean_map, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr', cluster_threshold=5)
        uncorr_map, threshold = threshold_stats_img(z_map, alpha=.001, height_control='fpr')

        # save FDR map
        print('Generating views')
        view = plotting.view_img(clean_map, threshold=3, title='{} contrast (FDR<0.05), Noyaux > 10 voxels'.format(contrast))
        view.save_as_html(op.join(figures_path, f'subject-level-from-{from_level}', f'{contrast}', f'{sub}_{contrast}_slm_FDRcluster_fwhm5.html'))
        plot_img_on_surf(clean_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast,f'{sub}_{contrast}_FDR.png'))
        plot_stat_map(clean_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast, f'{sub}_{contrast}_slices_FDR.png'))

        # save also uncorrected map
        view = plotting.view_img(uncorr_map, threshold=3, title='{} contrast (p<0.001), uncorr'.format(contrast))
        view.save_as_html(op.join(figures_path, f'subject-level-from-{from_level}', f'{contrast}', f'{sub}_{contrast}_slm_uncorr_fwhm5.html'))
        plot_img_on_surf(uncorr_map, bg_img=bg_img, vmax=6, output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast, f'{sub}_{contrast}_uncorr.png'))
        plot_stat_map(uncorr_map, bg_img=bg_img, vmax=6, display_mode='x', output_file=op.join(shinobi_behav.figures_path, 'subject-level-from-session', contrast, f'{sub}_{contrast}_slices_uncorr.png'))
        print('Done')
