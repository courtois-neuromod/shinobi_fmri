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
import os.path as op
from itertools import product
import numpy as np
import tqdm
import pickle
from joblib import Parallel, delayed
import nibabel as nib

def mem_used():
    tot = psutil.virtual_memory().total / 2**30
    used = psutil.virtual_memory().used / 2**30
    print('Memory currently used : {} Gis much more o/{} Go'.format(used, tot))


## Set constants
path_to_data = DATA_PATH
figures_path = FIG_PATH
contrasts = ['Kill', 'HealthLoss', 'HIT', 'JUMP', 'LEFT', 'RIGHT', 'DOWN']# + [f"{x}X{y}" for x,y in product(["HIT", "Kill", "HealthLoss"],["lvl1", "lvl4", "lvl5"])]
subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
results_path = '/home/hyruuk/scratch/neuromod/shinobi2024/processed/ses-level_beta_maps_ICC.pkl'
model = "simple"

# Create common masker
mask_files = []
for sub in subjects:
    mask_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin6Asym_desc-brain_mask.nii.gz",
    )
    shinobi_fname = op.join(
        path_to_data,
        "shinobi.fmriprep",
        sub,
        "ses-005",
        "func",
        f"{sub}_ses-005_task-shinobi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    # Load and resample (i.e. morph  ?) anat mask
    shinobi_raw_img = nib.load(shinobi_fname)
    aff_orig = shinobi_raw_img.affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    target_shape = shinobi_raw_img.shape[:3]
    mask_resampled = image.resample_img(mask_fname, target_affine=target_affine, target_shape=target_shape)
    mask_files.append(mask_resampled)
masker = NiftiMasker()
masker.fit(mask_files)
print('masks done')

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

# Load shinobi data
for contrast in contrasts:
    try:
        files = os.listdir(path_to_data + 'processed/beta_maps/ses-level/{}/'.format(contrast))
        for file in files:
            if model in file:
                fpath = path_to_data + 'processed/beta_maps/ses-level/{}/'.format(contrast) + file
                file_split = file.split('_')
                sub = file_split[0]
                ses = file_split[1]
                run = file[20]
                print(f'Loading {fpath}')
                raw_dpath = op.join(
                    path_to_data,
                    "shinobi.fmriprep",
                    sub,
                    ses,
                    "func",
                    f"{sub}_{ses}_task-shinobi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                )
                niimap = image.load_img(fpath)
                maps.append(masker.transform(niimap))
                subj_arr.append(sub)
                sess_arr.append(ses)
                #run_arr.append(run)
                cond_arr.append(contrast)
                #raw_data.append(image.concat_imgs(raw_dpath))
                fnames.append(raw_dpath)
                mapnames.append(fpath)
    except Exception as e:
        print(e)
        print('no file for contrast {}'.format(contrast))
        continue
    
# Load HCP data (treat HCP run as "sessions")
for sub in subjects:
    subfolder = op.join(path_to_data, "hcp_results", sub)
    runfolders = [f for f in os.listdir(subfolder) if 'run-' in f]
    for runfolder in runfolders:
        conditions = [f.split('.')[0] for f in os.listdir(op.join(subfolder, runfolder, 'effect_size_maps')) if '.nii.gz' in f]
        for cond in conditions:
            fpath = op.join(subfolder, runfolder, 'effect_size_maps', '{}.nii.gz'.format(cond))
            print(f'Loading {fpath}')
            niimap = image.load_img(fpath)
            resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
            maps.append(masker.transform(resampled_img))
            subj_arr.append(sub)
            sess_arr.append('_'.join(runfolder.split('_')[2:]))
            #run_arr.append(run)
            cond_arr.append(cond)
            #raw_data.append(image.concat_imgs(raw_dpath))
            fnames.append(raw_dpath)
            mapnames.append(fpath)



print('loading done')


# Create the corr_matrix files or load it if it already exists
if not os.path.isfile(results_path):
    corr_matrix = np.zeros((len(maps), len(maps)))
    dict = {'corr_matrix': corr_matrix,
            'fnames': fnames,
            'subj': subj_arr,
            'ses': sess_arr,
            #'run': run_arr,
            'cond': cond_arr}
    with open(results_path, 'wb') as f:
        pickle.dump(dict, f)

else:
    with open(results_path, 'rb') as f:
        dict = pickle.load(f)
        corr_matrix = np.array(dict['corr_matrix'])



### Compute the correlation coefficients
def compute_corrcoef(i, j, maps, corr_matrix):
    if j > i:
        if corr_matrix[i, j] == 0:
            print('Computing correlation between {} and {}'.format(i, j))
            coeff = np.corrcoef(maps[i], maps[j])[0, 1]
            return i, j, coeff
    return i, j, None

n_jobs = 1  # Set to the number of CPUs you want to use, -1 for all available CPUs
for i in tqdm.tqdm(range(len(maps))):
    # Parallelize the inner loop
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_corrcoef)(i, j, maps, np.array(corr_matrix)) for j in tqdm.tqdm(range(len(maps)))
    )

    # Update the corr_matrix with the new values
    for i, j, coeff in results:
        if coeff is not None:
            corr_matrix[i, j] = coeff

    
    # Save the results
    dict = {'corr_matrix': corr_matrix,
            'fnames': fnames,
            'subj': subj_arr,
            'ses': sess_arr,
            #'run': run_arr,
            'cond': cond_arr}
    with open(results_path, 'wb') as f:
        pickle.dump(dict, f)



fig, ax = plt.subplots(figsize=(15,15))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sbn.heatmap(corr_matrix, annot=True, xticklabels=mapnames, yticklabels=mapnames, ax=ax, cbar=False, mask=mask, annot_kws={"size":14})
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

if not os.path.isdir(figures_path + '/corrmats_withconstant/'):
    os.mkdir(figures_path + '/corrmats_withconstant/')
fig.savefig(figures_path + '/corrmats_withconstant/ses-level_corrmat.png')
