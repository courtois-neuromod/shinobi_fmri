from src.params import path_to_data, figures_path
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import image
import os

sub = 'sub-01'
sessions = ['ses-002', 'ses-003', 'ses-004', 'ses-005', 'ses-006', 'ses-007', 'ses-008']
contrasts = ['HealthLoss', 'Jump', 'Hit']

for idx_con, contrast in enumerate(contrasts):
    width_fig = 20
    fig = plt.figure(figsize=(width_fig, len(sessions)*3))

    for idx_ses, ses in enumerate(sessions):
        file = path_to_data + 'processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
        if idx_ses == 0:
            cut_coords = plotting.find_xyz_cut_coords(file, activation_threshold=0.1)
        plotting.plot_stat_map(
            file,
            display_mode="ortho",
            axes=plt.subplot(len(sessions), 2, idx_ses + 1),
            threshold=None,
            vmax=10,
            colorbar=False,
            draw_cross=False,
            title=ses,
            cmap="bwr",
            cut_coords=cut_coords)

    if not os.path.isdir(path_to_data + './reports/figures/session_plots'):
        os.mkdir(path_to_data + './reports/figures/session_plots')
    fig.savefig('./reports/figures/session-level_{}.png'.format(contrast))
