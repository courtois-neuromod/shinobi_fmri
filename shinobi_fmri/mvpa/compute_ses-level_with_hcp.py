import os
import os.path as op
import pandas as pd
import numpy as np
import shinobi_behav
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.plotting import plot_stat_map, show
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sbn
import matplotlib.pyplot as plt
import pickle
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiMasker
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# remove convergence warning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default=None,
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

# Initialize paths and variables
path_to_data = shinobi_behav.DATA_PATH
models = ["simple"]
model = "simple"
CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'Kill', 'HealthLoss']
contrasts = CONDS_LIST
if args.subject is not None:
    subjects = [args.subject]
else:
    subjects = shinobi_behav.SUBJECTS

def create_common_masker(path_to_data, subjects, masker_kwargs=None):
    """
    Create a common NiftiMasker by resampling subject-specific brain masks to a common space.
    """
    if masker_kwargs is None:
        masker_kwargs = {}

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
        # Load and resample anatomical mask to functional image space
        shinobi_raw_img = nib.load(shinobi_fname)
        aff_orig = shinobi_raw_img.affine[:, -1]
        target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
        target_shape = shinobi_raw_img.shape[:3]
        mask_resampled = image.resample_img(
            mask_fname,
            target_affine=target_affine,
            target_shape=target_shape
        )
        mask_files.append(mask_resampled)

    masker = NiftiMasker(**masker_kwargs)
    masker.fit(mask_files)
    return masker, target_affine, target_shape

all_subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
masker, target_affine, target_shape = create_common_masker(path_to_data, all_subjects)

for sub in subjects:
    mvpa_results_path = op.join(path_to_data, "processed", "mvpa_results_with_hcp")
    os.makedirs(mvpa_results_path, exist_ok=True)
    decoder_fname = f"{sub}_{model}_decoder.pkl"
    decoder_pkl_path = op.join(mvpa_results_path, decoder_fname)

    mask_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    anat_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
    )

    # Initialize variables
    decoder = None
    results_dict = {}
    if op.isfile(decoder_pkl_path):
        # Load existing results
        with open(decoder_pkl_path, 'rb') as f:
            results_dict = pickle.load(f)
        decoder = results_dict.get('decoder')
        contrast_label = results_dict.get('contrast_label')
        confusion_matrices_dict = results_dict.get('confusion_matrices_dict')
        actual_per_class_accuracies = results_dict.get('actual_per_class_accuracies')
        permuted_per_class_accuracies = results_dict.get('permuted_per_class_accuracies')
        completed_permutations = results_dict.get('completed_permutations', 0)
        permuted_labels_list = results_dict.get('permuted_labels_list')
        print(f"Loaded existing results for {sub} from {decoder_pkl_path}")
    else:
        print(f"No existing results found for {sub}, starting fresh.")
        completed_permutations = 0

    if decoder is None:
        # Prepare data
        z_maps = []
        contrast_label = []
        session_label = []

        for contrast in contrasts:
            z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
            for z_map_fname in os.listdir(z_maps_fpath):
                if model in z_map_fname and sub in z_map_fname:
                    print(f'Loading {z_map_fname}')
                    session = z_map_fname.split("_")[1]
                    niimap = image.load_img(op.join(z_maps_fpath, z_map_fname))
                    resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
                    z_maps.append(resampled_img)
                    contrast_label.append(contrast)
                    session_label.append(session)

        subfolder = op.join(path_to_data, "hcp_results", sub)
        runfolders = [f for f in os.listdir(subfolder) if 'run-' in f]
        for runfolder in runfolders:
            conditions = [f.split('.')[0] for f in os.listdir(op.join(subfolder, runfolder, 'z_score_maps')) if '.nii.gz' in f]
            for cond in conditions:
                fpath = op.join(subfolder, runfolder, 'z_score_maps', f'{cond}.nii.gz')
                print(f'Loading {fpath}')
                niimap = image.load_img(fpath)
                resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
                z_maps.append(resampled_img)
                session_label.append('_'.join(runfolder.split('_')[2:]))
                contrast_label.append(cond)

        # Fit the decoder on original data
        estimator = LinearSVC()#LogisticRegression(solver='saga', max_iter=100000)#LinearSVC(max_iter=1000, )
        decoder = Decoder(estimator=estimator, mask=masker, standardize=True, scoring='balanced_accuracy',
                          screening_percentile=10, cv=LeaveOneGroupOut(), n_jobs=16, verbose=1)
        decoder.fit(z_maps, contrast_label, groups=session_label)

        classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
        chance_level = 1. / len(np.unique(contrast_label))
        print(f'Decoding : {sub} {model}')
        print('Classification accuracy: {:.4f} / Chance level: {}'.format(
            classification_accuracy, chance_level))
        
        # Extract actual per-class accuracies
        actual_per_class_accuracies = {}
        for class_label in decoder.cv_scores_:
            actual_per_class_accuracies[class_label] = np.mean(decoder.cv_scores_[class_label])

        # Generate confusion matrices across folds
        confusion_matrices = []
        confusion_matrices_true_norm = []
        confusion_matrices_pred_norm = []
        confusion_matrices_all_norm = []
        for train, test in decoder.cv.split(z_maps, contrast_label, groups=session_label):
            decoder.fit(np.array(z_maps)[train], np.array(contrast_label)[train], groups=np.array(session_label)[train])
            y_pred = decoder.predict(np.array(z_maps)[test])
            y_true = np.array(contrast_label)[test]

            # Compute confusion matrices
            confusion_mat = confusion_matrix(y_true, y_pred, normalize=None, labels=decoder.classes_) 
            confusion_matrices.append(confusion_mat)
            confusion_mat_true_norm = confusion_matrix(y_true, y_pred, normalize='true', labels=decoder.classes_) 
            confusion_matrices_true_norm.append(confusion_mat_true_norm)
            confusion_mat_pred_norm = confusion_matrix(y_true, y_pred, normalize='pred', labels=decoder.classes_)
            confusion_matrices_pred_norm.append(confusion_mat_pred_norm)
            confusion_mat_all_norm = confusion_matrix(y_true, y_pred, normalize='all', labels=decoder.classes_)
            confusion_matrices_all_norm.append(confusion_mat_all_norm)
        confusion_matrices_dict = {
            'confusion_matrices': confusion_matrices,
            'confusion_matrices_true_norm': confusion_matrices_true_norm,
            'confusion_matrices_pred_norm': confusion_matrices_pred_norm,
            'confusion_matrices_all_norm': confusion_matrices_all_norm
        }

        # Initialize permutation testing variables
        n_permutations = 1000  # Set the number of permutations
        permuted_per_class_accuracies = {class_label: [] for class_label in np.unique(contrast_label)}
        completed_permutations = 0

        # Set random seed and generate all permutations upfront
        np.random.seed(42)  # Set a seed for reproducibility
        permuted_labels_list = []
        for perm in range(n_permutations):
            permuted_labels = np.random.permutation(contrast_label)
            permuted_labels_list.append(permuted_labels)
        print(f"Generated {n_permutations} permutations.")

        # Save initial state
        results_dict = {
            'decoder': decoder,
            'contrast_label': contrast_label,
            'confusion_matrices_dict': confusion_matrices_dict,
            'actual_per_class_accuracies': actual_per_class_accuracies,
            'permuted_per_class_accuracies': permuted_per_class_accuracies,
            'completed_permutations': completed_permutations,
            'permuted_labels_list': permuted_labels_list
        }
        with open(decoder_pkl_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Initial results saved for {sub}.")

    else:
        # If decoder is already loaded, ensure z_maps and session_label are loaded
        if 'z_maps' not in locals():
            # Prepare data (similar to above)
            z_maps = []
            session_label = []
            for contrast in contrasts:
                z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
                for z_map_fname in os.listdir(z_maps_fpath):
                    if model in z_map_fname and sub in z_map_fname:
                        session = z_map_fname.split("_")[1]
                        niimap = image.load_img(op.join(z_maps_fpath, z_map_fname))
                        resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
                        z_maps.append(resampled_img)
                        session_label.append(session)

            subfolder = op.join(path_to_data, "hcp_results", sub)
            runfolders = [f for f in os.listdir(subfolder) if 'run-' in f]
            for runfolder in runfolders:
                conditions = [f.split('.')[0] for f in os.listdir(op.join(subfolder, runfolder, 'z_score_maps')) if '.nii.gz' in f]
                for cond in conditions:
                    fpath = op.join(subfolder, runfolder, 'z_score_maps', f'{cond}.nii.gz')
                    niimap = image.load_img(fpath)
                    resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
                    z_maps.append(resampled_img)
                    session_label.append('_'.join(runfolder.split('_')[2:]))

    # Perform permutation testing
    n_permutations = 1  # Ensure this matches the initial setting
    print(f"Starting permutation testing for {sub}. Completed {completed_permutations}/{n_permutations} permutations so far.")

    try:
        for perm_index in range(completed_permutations, n_permutations):
            permuted_labels = permuted_labels_list[perm_index]
            # Initialize a new decoder with the same parameters
            estimator = LinearSVC()
            decoder_perm = Decoder(estimator=estimator, mask=masker, standardize=True, scoring='balanced_accuracy',
                            screening_percentile=10, cv=LeaveOneGroupOut(), n_jobs=16, verbose=1)
            # Fit decoder with permuted labels
            decoder_perm.fit(z_maps, permuted_labels, groups=session_label)
            # Extract per-class accuracies
            for class_label in decoder_perm.cv_scores_:
                permuted_accuracy = np.mean(decoder_perm.cv_scores_[class_label])
                permuted_per_class_accuracies[class_label].append(permuted_accuracy)

            completed_permutations += 1

            # Save progress after each permutation or every 10 permutations
            if completed_permutations % 10 == 0 or completed_permutations == n_permutations:
                # Update results_dict
                results_dict['permuted_per_class_accuracies'] = permuted_per_class_accuracies
                results_dict['completed_permutations'] = completed_permutations
                with open(decoder_pkl_path, 'wb') as f:
                    pickle.dump(results_dict, f)
                print(f"Saved progress after {completed_permutations}/{n_permutations} permutations.")

            if completed_permutations % 50 == 0:
                print(f"Completed {completed_permutations}/{n_permutations} permutations.")

    except KeyboardInterrupt:
        # Handle interruption and save progress
        results_dict['permuted_per_class_accuracies'] = permuted_per_class_accuracies
        results_dict['completed_permutations'] = completed_permutations
        with open(decoder_pkl_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Interrupted at permutation {completed_permutations}. Progress saved.")
        exit()

    # Compute p-values for each class
    p_values = {}
    for class_label in actual_per_class_accuracies:
        actual_accuracy = actual_per_class_accuracies[class_label]
        permuted_accuracies = permuted_per_class_accuracies[class_label]
        # Compute p-value as the proportion of permuted accuracies >= actual accuracy
        p_value = np.mean([acc >= actual_accuracy for acc in permuted_accuracies])
        p_values[class_label] = p_value

    # Print p-values
    print("Class-specific p-values from permutation testing:")
    for class_label, p_value in p_values.items():
        print(f"Class {class_label}: p-value = {p_value}")

    # Save final results
    results_dict['p_values'] = p_values
    with open(decoder_pkl_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Saved final results for {sub} to {decoder_pkl_path}")

    # Plot weights
    for cond in np.unique(contrast_label):
        output_fname = op.join("./", "reports", "figures", "ses-level", cond, "MVPA", f"{sub}_{cond}_{model}_mvpa.png")
        os.makedirs(op.join("./", "reports", "figures", "ses-level", cond, "MVPA"), exist_ok=True)
        weight_img = decoder.coef_img_[cond]
        plot_stat_map(weight_img, bg_img=anat_fname, title=f"SVM weights {cond}", output_file=output_fname)
        nib.save(weight_img, op.join(mvpa_results_path, f"{sub}_{cond}_{model}_mvpa.nii.gz"))