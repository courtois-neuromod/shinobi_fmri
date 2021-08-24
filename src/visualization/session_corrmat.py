from src.params import path_to_data, figures_path
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import image
import os
import numpy as np
import seaborn as sbn

## Set constants
sub = 'sub-01'
sessions = ['ses-002', 'ses-003', 'ses-004', 'ses-005', 'ses-006', 'ses-007', 'ses-008']
contrasts = ['HealthLoss', 'Jump', 'Hit']


maps = []
names = []
for contrast in contrasts:
    for ses in sessions:
        file = path_to_data + 'processed/cmaps/{}/{}_{}.nii.gz'.format(contrast, sub, ses)
        niimap = image.load_img(file)
        maps.append(niimap.get_fdata())
        names.append(contrast + '_' + ses)

maps = np.array(maps)
corr_map = np.zeros(maps.shape[1:])
sess_idx = [x for x in range(maps.shape[0])]

corr_matrix = np.zeros((len(maps), len(maps)))
for i in range(len(maps)):
    for j in range(len(maps)):
        imap = maps[i,:,:,:].squeeze().flatten()
        imap_trim = np.array([x for x in imap if x != 0])
        jmap = maps[j,:,:,:].squeeze().flatten()
        jmap_trim = np.array([x for x in jmap if x != 0])
        corr_matrix[i,j] = np.corrcoef(imap_trim, jmap_trim)[0,1]

fig, ax = plt.subplots(figsize=(15,15))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sbn.heatmap(corr_matrix, annot=True, xticklabels=names, yticklabels=names, ax=ax, cbar=False, mask=mask, annot_kws={"size":14, ''})
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

if not os.path.isdir(path_to_data + './reports/figures/session_plots'):
    os.mkdir(path_to_data + './reports/figures/session_plots')
fig.savefig('./reports/figures/session-level_{}.png'.format(contrast))
