from shinobi_behav import path_to_data, figures_path
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import image
import os
import numpy as np
import seaborn as sbn
from nilearn.input_data import NiftiMasker

## Set constants
contrasts = ['HealthLoss', 'Jump', 'Hit']
subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']

path_to_data = '/home/hyruuk/GitHub/neuromod/shinobi_fmri/data/'


# load data
maps = []
raw_data = []
subj_arr = []
sess_arr = []
run_arr = []
cond_arr = []
fnames = []
for contrast in contrasts:
    files = os.listdir(path_to_data + 'processed/cmaps/run-level/{}/'.format(contrast))
    for file in files[]:
        fpath = path_to_data + 'processed/cmaps/run-level/{}/'.format(contrast) + file
        sub = file[0:6]
        ses = file[7:14]
        run = file[20]
        print('run : '+ file)
        raw_dpath = '/media/storage/neuromod/shinobi_data/shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        niimap = image.load_img(fpath)
        maps.append(niimap.get_fdata())
        subj_arr.append(sub)
        sess_arr.append(ses)
        run_arr.append(run)
        cond_arr.append(contrast)
        #raw_data.append(image.concat_imgs(raw_dpath))
        fnames.append(fpath)

# Create common masker
masker = NiftiMasker()
masker.fit(fnames)

masked_maps = []
for map in maps:
    masked_maps.append(masker.transform(map))

maps = np.array(maps)
corr_map = np.zeros(maps.shape[1:])
sess_idx = [x for x in range(maps.shape[0])]

# compute corrcoefs
corr_matrix = np.zeros((len(maps), len(maps)))
for i in range(len(maps)):
    for j in range(len(maps)):
        # if j>i
        imap = maps[i,:,:,:].squeeze().flatten()
        imap_trim = np.array([x for x in imap if x != 0])
        jmap = maps[j,:,:,:].squeeze().flatten()
        jmap_trim = np.array([x for x in jmap if x != 0])
        print('i : {}, j : {}'.format(i,j))
        print('n zeros in jth map : {}'.format(np.unique(np.unique(jmap, return_counts=True)[1])[1]))
        print('isize : {}, jsize : {}'.format(imap_trim.shape, jmap_trim.shape))
        corr_matrix[i,j] = np.corrcoef(imap_trim, jmap_trim)[0,1]

dict = {'corr_matrix': corr_matrix,
        'fnames': fnames,
        'subj': subj_arr,
        'ses': sess_arr,
        'run': run_arr,
        'cond': cond_arr}

results_path = '/home/hyruuk/scratch/neuromod/shinobi_data/processed/cmaps/runlevel_maps_corrs.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(f, dict)

'''
fig, ax = plt.subplots(figsize=(15,15))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sbn.heatmap(corr_matrix, annot=True, xticklabels=names, yticklabels=names, ax=ax, cbar=False, mask=mask, annot_kws={"size":14, ''})
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

if not os.path.isdir(path_to_data + './reports/figures/session_plots'):
    os.mkdir(path_to_data + './reports/figures/session_plots')
fig.savefig('./reports/figures/session-level_{}.png'.format(contrast))
'''
