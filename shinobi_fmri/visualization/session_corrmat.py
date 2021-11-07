from shinobi_behav import path_to_data, figures_path
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import image
import os
import numpy as np
import seaborn as sbn
from nilearn.input_data import NiftiMasker
import psutil
import pickle
import tqdm

def mem_used():
    tot = psutil.virtual_memory().total / 10**9
    used = psutil.virtual_memory().used / 10**9
    print('Memory currently used : {} Go/{} Go'.format(used, tot))


## Set constants
contrasts = ['HealthLoss', 'Jump', 'Hit']
subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']

#path_to_data = '/home/hyruuk/scratch/neuromod/shinobi_data/'
mem_used()

# load data
maps = []
raw_data = []
subj_arr = []
sess_arr = []
run_arr = []
cond_arr = []
fnames = []
mapnames = []
for contrast in contrasts:
    files = os.listdir(path_to_data + 'processed/cmaps/run-level/{}/'.format(contrast))
    for file in files:
        fpath = path_to_data + 'processed/cmaps/run-level/{}/'.format(contrast) + file
        sub = file[0:6]
        ses = file[7:14]
        run = file[20]
        print('run : '+ file)
        raw_dpath = path_to_data + 'shinobi/derivatives/fmriprep-20.2lts/fmriprep/{}/{}/func/{}_{}_task-shinobi_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, ses, sub, ses, run)
        niimap = image.load_img(fpath)
        maps.append(niimap)
        subj_arr.append(sub)
        sess_arr.append(ses)
        run_arr.append(run)
        cond_arr.append(contrast)
        #raw_data.append(image.concat_imgs(raw_dpath))
        fnames.append(raw_dpath)
        mapnames.append(fpath)


print('loading done')
mem_used()
mask_fnames = []
for sub in subjects:
    list = [x for x in fnames if sub in x]
    mask_fnames.append(list[0])

# Create common masker
masker = NiftiMasker()
masker.fit(mask_fnames)
print('masks done')
mem_used()
masked_maps = []
for map in maps:
    masked_maps.append(masker.transform(map))
print('transforms done')
mem_used()

maps = np.array(masked_maps).squeeze()

#sess_idx = [x for x in range(maps.shape[0])]

# compute corrcoefs
corr_matrix = np.zeros((len(maps), len(maps)))
for i in tqdm.tqdm(range(len(maps))):
    for j in tqdm.tqdm(range(len(maps))):
        if j>i:
            imap = maps[i]
            imap_trim = np.array([x for x in imap if x != 0])
            jmap = maps[j]
            jmap_trim = np.array([x for x in jmap if x != 0])
            coeff = np.corrcoef(imap_trim, jmap_trim)[0,1]
            corr_matrix[i,j] = coeff


dict = {'corr_matrix': corr_matrix,
        'fnames': fnames,
        'subj': subj_arr,
        'ses': sess_arr,
        'run': run_arr,
        'cond': cond_arr}

results_path = '/home/hyruuk/scratch/neuromod/shinobi_data/processed/cmaps/runlevel_maps_corrs.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(dict, f)


fig, ax = plt.subplots(figsize=(15,15))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sbn.heatmap(corr_matrix, annot=True, xticklabels=mapnames, yticklabels=mapnames, ax=ax, cbar=False, mask=mask, annot_kws={"size":14})
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

if not os.path.isdir(figures_path + '/run_corrmats/'):
    os.mkdir(figures_path + '/run_corrmats/')
fig.savefig(figures_path + '/run_corrmats/run-level_corrmat.png')
