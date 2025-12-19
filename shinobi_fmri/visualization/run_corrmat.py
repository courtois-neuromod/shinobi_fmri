from shinobi_fmri.config import DATA_PATH, FIG_PATH
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
import numpy as np
import tqdm
import pickle
from joblib import Parallel, delayed


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
        # New structure: processed/run-level/sub-XX/ses-YY/z_maps/
        run_level_dir = op.join(path_to_data, 'processed', 'run-level')

        if not op.exists(run_level_dir):
            print(f'Run-level directory not found: {run_level_dir}')
            continue

        # Iterate through subjects
        for sub in subjects:
            sub_dir = op.join(run_level_dir, sub)
            if not op.exists(sub_dir):
                continue

            # Iterate through sessions
            for ses_name in os.listdir(sub_dir):
                if not ses_name.startswith('ses-'):
                    continue

                z_maps_dir = op.join(sub_dir, ses_name, 'z_maps')
                if not op.exists(z_maps_dir):
                    continue

                # Look for z-maps with this contrast
                for file in os.listdir(z_maps_dir):
                    if f'contrast-{contrast}' in file and file.endswith('stat-z.nii.gz'):
                        # Parse filename: sub-XX_ses-YY_task-shinobi_run-XX_contrast-COND_stat-z.nii.gz
                        file_parts = file.split('_')
                        run_part = [p for p in file_parts if 'run-' in p]
                        if run_part:
                            run = run_part[0]  # e.g., 'run-01'

                            fpath = op.join(z_maps_dir, file)
                            print(f'Loading: {file}')

                            raw_dpath = op.join(
                                path_to_data,
                                "shinobi.fmriprep",
                                sub,
                                ses_name,
                                "func",
                                f"{sub}_{ses_name}_task-shinobi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                            )
                            niimap = image.load_img(fpath)
                            maps.append(niimap)
                            subj_arr.append(sub)
                            sess_arr.append(ses_name)
                            run_arr.append(run)
                            cond_arr.append(contrast)
                            fnames.append(raw_dpath)
                            mapnames.append(fpath)
    except Exception as e:
        print(e)
        print(f'Error processing contrast {contrast}')
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
        corr_matrix = np.array(dict['corr_matrix'])

def compute_corrcoef(i, j, maps, corr_matrix):
    if j > i:
        if corr_matrix[i, j] == 0:
            imap = maps[i]
            imap_trim = np.array([x for x in imap if x != 0])
            jmap = maps[j]
            jmap_trim = np.array([x for x in jmap if x != 0])
            coeff = np.corrcoef(imap_trim, jmap_trim)[0, 1]
            return i, j, coeff
    return i, j, None

n_jobs = -1  # Set to the number of CPUs you want to use, -1 for all available CPUs
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
