from shinobi_behav import DATA_PATH, FIG_PATH
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
    tot = psutil.virtual_memory().total / 2**30
    used = psutil.virtual_memory().used / 2**30
    print('Memory currently used : {} Go/{} Go'.format(used, tot))


## Set constants
path_to_data = DATA_PATH
figures_path = FIG_PATH
contrasts = ['Kill', 'JUMP', 'HIT', 'LEFT', 'RIGHT', 'DOWN', 'HealthGain', 'HealthLoss', 'RIGHT+LEFT+DOWN', 'HIT+JUMP']
subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
results_path = '/home/hyruuk/scratch/neuromod/shinobi2023/processed/run-level_maps_corrs.pkl'

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
    if contrast in ['RIGHT+LEFT+DOWN', 'HIT+JUMP']:
        model = "intermediate"
    else:
        model = "simple"
    try:
        files = os.listdir(path_to_data + 'processed/z_maps/run-level/{}/'.format(contrast))
        for file in files:
            if model in file:
                fpath = path_to_data + 'processed/z_maps/run-level/{}/'.format(contrast) + file
                file_split = file.split('_')
                sub = file_split[0]
                ses = file_split[1]
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
    except Exception as e:
        print(e)
        print('no file for contrast {}'.format(contrast))
        continue


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
for map in tqdm.tqdm(maps):
    masked_maps.append(masker.transform(map))
print('transforms done')
mem_used()

maps = np.array(masked_maps).squeeze()

# Create the corr_matrix files or load it if it already exists
if not os.path.isfile(results_path):
    corr_matrix = np.zeros((len(maps), len(maps)))
    dict = {'corr_matrix': corr_matrix,
            'fnames': fnames,
            'subj': subj_arr,
            'ses': sess_arr,
            'run': run_arr,
            'cond': cond_arr}
    with open(results_path, 'wb') as f:
        pickle.dump(dict, f)

else:
    with open(results_path, 'rb') as f:
        dict = pickle.load(f)
        corr_matrix = dict['corr_matrix']

# compute corrcoefs
for i in tqdm.tqdm(range(len(maps))):
    for j in tqdm.tqdm(range(len(maps))):
        if j>i:
            if corr_matrix[i,j] == 0:
                imap = maps[i]
                imap_trim = np.array([x for x in imap if x != 0])
                jmap = maps[j]
                jmap_trim = np.array([x for x in jmap if x != 0])
                coeff = np.corrcoef(imap_trim, jmap_trim)[0,1]
                corr_matrix[i,j] = coeff
                # immediately saves the result
                dict = {'corr_matrix': corr_matrix,
                        'fnames': fnames,
                        'subj': subj_arr,
                        'ses': sess_arr,
                        'run': run_arr,
                        'cond': cond_arr}
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