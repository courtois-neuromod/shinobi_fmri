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
from joblib import Parallel, delayed
import nibabel as nib


def mem_used():
    """Print current memory usage."""
    tot = psutil.virtual_memory().total / 2**30
    used = psutil.virtual_memory().used / 2**30
    print(f'Memory currently used : {used:.2f} Go / {tot:.2f} Go')


def create_masker(subjects, path_to_data):
    """Create a common NiftiMasker from subject masks."""
    mask_files = []
    target_affine = None
    target_shape = None
    
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
        
        # Load and resample anat mask
        shinobi_raw_img = nib.load(shinobi_fname)
        aff_orig = shinobi_raw_img.affine[:, -1]
        target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
        target_shape = shinobi_raw_img.shape[:3]
        
        mask_resampled = image.resample_img(
            mask_fname, 
            target_affine=target_affine, 
            target_shape=target_shape, 
            force_resample=True, 
            copy_header=True
        )
        mask_files.append(mask_resampled)
    
    masker = NiftiMasker()
    masker.fit(mask_files)
    print('Masks created')
    
    return masker, target_affine, target_shape


def load_shinobi_data(contrasts, path_to_data, model, masker):
    """Load Shinobi beta maps."""
    maps = []
    subj_arr = []
    sess_arr = []
    cond_arr = []
    fnames = []
    mapnames = []
    
    for contrast in contrasts:
        try:
            contrast_dir = op.join(path_to_data, 'processed/beta_maps/ses-level', contrast)
            files = os.listdir(contrast_dir)
            
            for file in files:
                if model in file:
                    fpath = op.join(contrast_dir, file)
                    file_split = file.split('_')
                    sub = file_split[0]
                    ses = file_split[1]
                    
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
                    cond_arr.append(contrast)
                    fnames.append(raw_dpath)
                    mapnames.append(fpath)
                    
        except Exception as e:
            print(f'Error loading contrast {contrast}: {e}')
            continue
    
    return maps, subj_arr, sess_arr, cond_arr, fnames, mapnames


def load_hcp_data(subjects, path_to_data, masker, target_affine, target_shape):
    """Load HCP effect size maps."""
    maps = []
    subj_arr = []
    sess_arr = []
    cond_arr = []
    fnames = []
    mapnames = []
    
    for sub in subjects:
        subfolder = op.join(path_to_data, "hcp_results", sub)
        runfolders = [f for f in os.listdir(subfolder) if 'run-' in f]
        
        for runfolder in runfolders:
            effect_size_dir = op.join(subfolder, runfolder, 'effect_size_maps')
            conditions = [f.split('.')[0] for f in os.listdir(effect_size_dir) if '.nii.gz' in f]
            
            for cond in conditions:
                fpath = op.join(effect_size_dir, f'{cond}.nii.gz')
                print(f'Loading {fpath}')
                
                niimap = image.load_img(fpath)
                resampled_img = image.resample_img(
                    niimap, 
                    target_affine=target_affine, 
                    target_shape=target_shape, 
                    force_resample=True, 
                    copy_header=True
                )
                
                maps.append(masker.transform(resampled_img))
                subj_arr.append(sub)
                sess_arr.append('_'.join(runfolder.split('_')[2:]))
                cond_arr.append(cond)
                fnames.append('')  # No raw data path for HCP
                mapnames.append(fpath)
    
    return maps, subj_arr, sess_arr, cond_arr, fnames, mapnames


def load_or_create_corr_matrix(results_path, num_maps, fnames, subj_arr, sess_arr, cond_arr):
    """Load existing correlation matrix or create a new one."""
    if not os.path.isfile(results_path):
        corr_matrix = np.zeros((num_maps, num_maps))
        data_dict = {
            'corr_matrix': corr_matrix,
            'fnames': fnames,
            'subj': subj_arr,
            'ses': sess_arr,
            'cond': cond_arr
        }
        with open(results_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print('Created new correlation matrix')
    else:
        with open(results_path, 'rb') as f:
            data_dict = pickle.load(f)
            corr_matrix = np.array(data_dict['corr_matrix'])
            
            # Ensure corr_matrix has correct size if new maps were added
            if corr_matrix.shape[0] != num_maps:
                old_size = corr_matrix.shape[0]
                new_corr_matrix = np.zeros((num_maps, num_maps))
                new_corr_matrix[:old_size, :old_size] = corr_matrix
                corr_matrix = new_corr_matrix
                print(f'Resized correlation matrix from {old_size} to {num_maps}')
        print('Loaded existing correlation matrix')
    
    return corr_matrix


def compute_corrcoef(i, j, maps, corr_matrix):
    """Compute correlation coefficient between two maps."""
    if j > i:
        if corr_matrix[i, j] == 0:
            print(f'Computing correlation between {i} and {j}')
            coeff = np.corrcoef(maps[i], maps[j])[0, 1]
            return i, j, coeff
    return i, j, None


def compute_correlations(maps, corr_matrix, results_path, fnames, subj_arr, sess_arr, cond_arr, n_jobs=1):
    """Compute all pairwise correlations between maps."""
    for i in tqdm.tqdm(range(len(maps)), desc='Computing correlations'):
        # Parallelize the inner loop
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_corrcoef)(i, j, maps, np.array(corr_matrix)) 
            for j in range(len(maps))
        )
        
        # Update the corr_matrix with the new values
        for idx_i, idx_j, coeff in results:
            if coeff is not None:
                corr_matrix[idx_i, idx_j] = coeff
        
        # Save the results after each row
        data_dict = {
            'corr_matrix': corr_matrix,
            'fnames': fnames,
            'subj': subj_arr,
            'ses': sess_arr,
            'cond': cond_arr
        }
        with open(results_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    return corr_matrix


def plot_correlation_matrix(corr_matrix, mapnames, figures_path):
    """Create and save heatmap of correlation matrix."""
    fig, ax = plt.subplots(figsize=(15, 15))
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    sbn.heatmap(
        corr_matrix, 
        annot=True, 
        xticklabels=mapnames, 
        yticklabels=mapnames, 
        ax=ax, 
        cbar=False, 
        mask=mask, 
        annot_kws={"size": 14}
    )
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    
    output_dir = op.join(figures_path, 'corrmats_withconstant')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    output_path = op.join(output_dir, 'ses-level_corrmat.png')
    fig.savefig(output_path)
    print(f'Saved correlation matrix plot to {output_path}')


def main():
    """Main function to run the correlation analysis."""
    # Set constants
    path_to_data = DATA_PATH
    figures_path = FIG_PATH
    contrasts = ['Kill', 'HealthLoss', 'HIT', 'JUMP', 'LEFT', 'RIGHT', 'DOWN']
    subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
    results_path = '/scratch/hyruuk/neuromod/shinobi2025/processed/ses-level_beta_maps_ICC.pkl'
    model = "simple"
    n_jobs = 1
    
    # Create masker
    masker, target_affine, target_shape = create_masker(subjects, path_to_data)
    mem_used()
    
    # Load Shinobi data
    print('\nLoading Shinobi data...')
    shinobi_data = load_shinobi_data(contrasts, path_to_data, model, masker)
    maps, subj_arr, sess_arr, cond_arr, fnames, mapnames = shinobi_data
    
    # Load HCP data
    print('\nLoading HCP data...')
    hcp_data = load_hcp_data(subjects, path_to_data, masker, target_affine, target_shape)
    
    # Combine data
    maps.extend(hcp_data[0])
    subj_arr.extend(hcp_data[1])
    sess_arr.extend(hcp_data[2])
    cond_arr.extend(hcp_data[3])
    fnames.extend(hcp_data[4])
    mapnames.extend(hcp_data[5])
    
    print(f'\nTotal maps loaded: {len(maps)}')
    mem_used()
    
    # Load or create correlation matrix
    corr_matrix = load_or_create_corr_matrix(
        results_path, len(maps), fnames, subj_arr, sess_arr, cond_arr
    )
    
    # Compute correlations
    print('\nComputing correlations...')
    corr_matrix = compute_correlations(
        maps, corr_matrix, results_path, fnames, subj_arr, sess_arr, cond_arr, n_jobs
    )
    
    # Plot results
    print('\nPlotting correlation matrix...')
    plot_correlation_matrix(corr_matrix, mapnames, figures_path)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
