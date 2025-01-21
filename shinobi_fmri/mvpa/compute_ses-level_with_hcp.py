import warnings
warnings.filterwarnings("ignore")
import os
import os.path as op
import numpy as np
import pickle
import argparse

import shinobi_behav
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.decoding import Decoder

from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import parallel_backend, Parallel, delayed

##############################################################################
# ARGUMENT PARSING
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", default=None, type=str,
                    help="Subject to process (e.g. 'sub-01').")
parser.add_argument("--task", choices=["classif", "perm"], default="classif",
                    help="Task to run: 'classif' for real classification, 'perm' for permutations.")
parser.add_argument("--perm-index", type=int, default=None,
                    help="Starting permutation index. If not given, we start from the lowest missing index.")
args = parser.parse_args()

##############################################################################
# GLOBAL CONFIG
##############################################################################
path_to_data = shinobi_behav.DATA_PATH
model = "simple"
CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'Kill', 'HealthLoss']
if args.subject is not None:
    subjects = [args.subject]
else:
    subjects = shinobi_behav.SUBJECTS

screening_percentile = 20
n_permutations = 1000
n_jobs = 12

##############################################################################
# HELPER FUNCTIONS
##############################################################################
def create_common_masker(path_to_data, subjects, masker_kwargs=None):
    """
    Create a common NiftiMasker by resampling subject-specific brain masks 
    to a common space.
    """
    if masker_kwargs is None:
        masker_kwargs = {}

    print("Creating a common NiftiMasker. Resampling subject-specific brain masks...")

    mask_files = []
    for sub in tqdm(subjects, desc="Common Masker Subjects"):
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


def load_zmaps_for_subject(sub, model, contrasts, path_to_data, target_affine, target_shape):
    """
    Load z-maps for a given subject from both the main z_maps folder 
    and the hcp_results folder.
    Returns:
        z_maps (list of Nifti images)
        contrast_label (list of str)
        session_label (list of str)
    """
    print(f"[{sub}] Loading z-maps from local and HCP-based results...")
    z_maps = []
    contrast_label = []
    session_label = []

    # 1) Load from main z_maps folder
    for contrast in contrasts:
        z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
        if not op.exists(z_maps_fpath):
            continue

        file_list = os.listdir(z_maps_fpath)
        for z_map_fname in tqdm(file_list, desc=f"Shinobi z_maps ({sub} {contrast})", leave=False):
            if model in z_map_fname and sub in z_map_fname:
                session = z_map_fname.split("_")[1]
                niimap = image.load_img(op.join(z_maps_fpath, z_map_fname))
                # Resample the image to the same shape+affine for your common mask
                resampled_img = image.resample_img(
                    niimap,
                    target_affine=target_affine,
                    target_shape=target_shape
                )
                z_maps.append(resampled_img)
                contrast_label.append(contrast)
                session_label.append(session)

    # 2) Load from HCP-based results
    subfolder = op.join(path_to_data, "hcp_results", sub)
    if op.exists(subfolder):
        runfolders = [f for f in os.listdir(subfolder) if 'run-' in f]
        for runfolder in tqdm(runfolders, desc=f"HCP z_maps ({sub})", leave=False):
            path_z = op.join(subfolder, runfolder, 'z_score_maps')
            if not op.exists(path_z):
                continue
            conditions = [
                f.split('.')[0] for f in os.listdir(path_z) 
                if f.endswith('.nii.gz')
            ]
            for cond in conditions:
                fpath = op.join(path_z, f'{cond}.nii.gz')
                if op.exists(fpath):
                    niimap = image.load_img(fpath)
                    resampled_img = image.resample_img(
                        niimap,
                        target_affine=target_affine,
                        target_shape=target_shape
                    )
                    z_maps.append(resampled_img)
                    session_label.append('_'.join(runfolder.split('_')[2:]))
                    contrast_label.append(cond)

    print(f"[{sub}] Loaded {len(z_maps)} z-maps total.")
    return z_maps, contrast_label, session_label


def compute_crossval_confusions_and_accuracies(X_data, y_data, group_labels, estimator_cls=LinearSVC, n_jobs=1):
    """
    Manually implement cross-validation (e.g., LeaveOneGroupOut) *in parallel* 
    to get:
      - A confusion matrix for each fold
      - Per-label accuracies across folds
    """
    cv = LeaveOneGroupOut()
    splits = list(cv.split(X_data, y_data, groups=group_labels))
    all_labels = np.unique(y_data)

    def _fit_and_predict_fold(train_idx, test_idx):
        """Helper for each fold."""
        clf = estimator_cls()
        clf.fit(X_data[train_idx], y_data[train_idx])
        y_pred = clf.predict(X_data[test_idx])
        cm = confusion_matrix(y_data[test_idx], y_pred, labels=all_labels)
        # Per-label accuracy in this fold
        fold_acc_dict = {}
        for lbl in all_labels:
            lbl_mask = (y_data[test_idx] == lbl)
            if np.sum(lbl_mask) > 0:
                fold_acc_dict[lbl] = np.mean(y_pred[lbl_mask] == lbl)
        return cm, fold_acc_dict

    # Parallelize folds
    with parallel_backend('threading'):
        with tqdm_joblib(tqdm(desc="Manual CV folds", total=len(splits))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_and_predict_fold)(train_idx, test_idx)
                for (train_idx, test_idx) in splits
            )

    # Collect fold confusion matrices & accuracies
    fold_confusions = []
    per_class_accuracies = {lbl: [] for lbl in all_labels}
    for cm, fold_acc_dict in results:
        fold_confusions.append(cm)
        for lbl, acc in fold_acc_dict.items():
            per_class_accuracies[lbl].append(acc)

    # Mean accuracy across folds for each label
    actual_per_class_accuracies = {
        lbl: np.mean(per_class_accuracies[lbl]) for lbl in all_labels
    }
    return fold_confusions, actual_per_class_accuracies


##############################################################################
# MAIN SCRIPT
##############################################################################
def main():
    # 1) Build or reuse a common masker
    all_subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
    masker, target_affine, target_shape = create_common_masker(path_to_data, all_subjects)

    for sub in subjects:
        print(f"\n===== Processing {sub} ===== (Task: {args.task})")
        mvpa_results_path = op.join(path_to_data, "processed", "mvpa_results_with_hcp")
        os.makedirs(mvpa_results_path, exist_ok=True)
        decoder_fname = f"{sub}_{model}_decoder.pkl"
        decoder_pkl_path = op.join(mvpa_results_path, decoder_fname)

        # 2) Load existing results if any
        if op.isfile(decoder_pkl_path):
            with open(decoder_pkl_path, 'rb') as f:
                results_dict = pickle.load(f)
            print(f"Loaded existing results for {sub} from {decoder_pkl_path}.")
        else:
            print(f"No existing results found for {sub}, starting fresh.")
            results_dict = {}

        contrast_label = results_dict.get('contrast_label', None)
        completed_permutations = results_dict.get('completed_permutations', 0)
        permutations_data = results_dict.get('permutations', {})
        confusion_matrices_dict = results_dict.get('confusion_matrices_dict', {})
        actual_per_class_accuracies = results_dict.get('actual_per_class_accuracies', None)

        # 3) Load subject data (list of nibabel images, plus labels)
        z_maps, contrast_label_loaded, session_label = load_zmaps_for_subject(
            sub, model, CONDS_LIST, path_to_data, target_affine, target_shape
        )

        # We also want a 2D array for manual CV
        X_niimgs = masker.transform(z_maps)

        # If we never had a label list, use the newly loaded
        if not contrast_label:
            contrast_label = contrast_label_loaded

        # --------------------------------------------------------------------
        # TASK = "classif": 
        #    - Manual CV => get fold-level confusion matrices
        #    - Single final Decoder => get weight maps
        # --------------------------------------------------------------------
        if args.task == "classif":
            print(f"[{sub}] Running manual cross-validation to get fold-wise confusion...")

            fold_confusions, actual_per_class_accuracies = compute_crossval_confusions_and_accuracies(
                X_niimgs, np.array(contrast_label), np.array(session_label),
                estimator_cls=LinearSVC, n_jobs=n_jobs
            )

            # Store fold-level confusion matrices
            confusion_matrices_dict = {'fold_confusions': fold_confusions}

            print("[Per-class accuracy across folds]:")
            for lbl in actual_per_class_accuracies:
                print(f"  {lbl}: {actual_per_class_accuracies[lbl]:.4f}")

            # 2) Fit a final Decoder on all data to get weight maps
            print(f"[{sub}] Training final Decoder on entire dataset to get weight maps...")
            # Here, pass the *list of nibabel images* plus the *masker*:
            final_decoder = Decoder(
                estimator=LinearSVC(),
                mask=masker,  # essential to compute weight maps as Nifti images
                standardize=True,
                scoring='balanced_accuracy',
                screening_percentile=screening_percentile,
                cv=None,  # no internal CV for the final model
                n_jobs=n_jobs,
                verbose=0
            )
            final_decoder.fit(z_maps, contrast_label)  # pass images, not X_niimgs

            # final_decoder.coef_img_ now holds the weight maps. 
            # e.g. final_decoder.coef_img_['HIT'] is a nibabel image

            # If permutations_data doesn't exist, we initialize it
            class_list = np.unique(contrast_label)
            n_classes = len(class_list)
            if not permutations_data:
                permutations_data = {
                    'class_labels': list(class_list),
                    'accuracies': np.full((n_permutations, n_classes), np.nan)
                }

            # Save everything
            results_dict.update({
                'decoder': final_decoder,
                'contrast_label': contrast_label,
                'confusion_matrices_dict': confusion_matrices_dict,
                'actual_per_class_accuracies': actual_per_class_accuracies,
                'permutations': permutations_data,
                'completed_permutations': completed_permutations
            })

            with open(decoder_pkl_path, 'wb') as f:
                pickle.dump(results_dict, f)
            print(f"[{sub}] Classification results saved to {decoder_pkl_path}.")

            for class_lbl in final_decoder.classes_:
                w_img = final_decoder.coef_img_[class_lbl]
                out_fname = f"{sub}_{class_lbl}_{model}_weights.nii.gz"
                nib.save(w_img, out_fname)

        # --------------------------------------------------------------------
        # TASK = "perm": do permutations from a starting index or the lowest missing
        # --------------------------------------------------------------------
        elif args.task == "perm":
            if 'permutations' not in results_dict or not actual_per_class_accuracies:
                raise RuntimeError(
                    f"[{sub}] No classification info found in {decoder_pkl_path}. "
                    "Run --task classif first."
                )

            perm_array = permutations_data['accuracies']  # shape: (n_permutations, n_classes)
            class_list = permutations_data['class_labels']

            if args.perm_index is not None:
                current_index = args.perm_index
            else:
                # find the first missing
                missing_indices = np.where(np.isnan(perm_array[:, 0]))[0]
                if len(missing_indices) == 0:
                    print(f"[{sub}] All permutations completed! Nothing to do.")
                    continue
                current_index = missing_indices[0]

            print(f"[{sub}] Starting permutations at index={current_index} (up to {n_permutations-1})...")

            try:
                while current_index < n_permutations:
                    # Skip if already done
                    if not np.isnan(perm_array[current_index, 0]):
                        print(f"[{sub}] Perm {current_index} is already computed. Skipping.")
                        current_index += 1
                        continue

                    print(f"[{sub}] Computing permutation {current_index}...")

                    # Make perm labels
                    np.random.seed(42 + current_index)
                    perm_labels = np.random.permutation(contrast_label)

                    # Use nilearn Decoder with CV in one shot
                    decoder_perm = Decoder(
                        estimator=LinearSVC(),
                        mask=masker,  # pass the same mask used for real classification
                        standardize=True,
                        scoring='balanced_accuracy',
                        screening_percentile=screening_percentile,
                        cv=LeaveOneGroupOut(),
                        n_jobs=n_jobs,
                        verbose=0
                    )
                    with parallel_backend('threading'):
                        with tqdm_joblib(tqdm(desc=f"[{sub}] Perm {current_index} CV Progress", leave=False)):
                            decoder_perm.fit(z_maps, perm_labels, groups=session_label)

                    # Store average CV accuracy for each class
                    for i_class, c_lbl in enumerate(class_list):
                        if c_lbl not in decoder_perm.cv_scores_:
                            perm_array[current_index, i_class] = np.nan
                        else:
                            perm_array[current_index, i_class] = np.mean(decoder_perm.cv_scores_[c_lbl])

                    # Save
                    permutations_data['accuracies'] = perm_array
                    completed_permutations = np.sum(~np.isnan(perm_array[:, 0]))
                    results_dict['completed_permutations'] = completed_permutations
                    results_dict['permutations'] = permutations_data

                    with open(decoder_pkl_path, 'wb') as f:
                        pickle.dump(results_dict, f)

                    print(f"[{sub}] => Perm {current_index} done. "
                          f"{completed_permutations}/{n_permutations} completed.")

                    # Check if we finished them all
                    if completed_permutations == n_permutations:
                        print(f"[{sub}] All permutations done! Computing p-values per class...")
                        p_values = {}
                        for i_class, c_lbl in enumerate(class_list):
                            real_acc = actual_per_class_accuracies[c_lbl]
                            perms_acc = perm_array[:, i_class]
                            p_val = np.mean(perms_acc >= real_acc)
                            p_values[c_lbl] = p_val
                        results_dict['p_values'] = p_values

                        with open(decoder_pkl_path, 'wb') as f:
                            pickle.dump(results_dict, f)

                        print("[Final p-values]:")
                        for c_lbl, val in p_values.items():
                            print(f"  {c_lbl}: p={val:.4f}")
                        break
                    else:
                        current_index += 1

            except KeyboardInterrupt:
                print(f"[{sub}] Interrupted at permutation {current_index}. Saving progress...")
                permutations_data['accuracies'] = perm_array
                results_dict['permutations'] = permutations_data
                results_dict['completed_permutations'] = np.sum(~np.isnan(perm_array[:, 0]))

                with open(decoder_pkl_path, 'wb') as f:
                    pickle.dump(results_dict, f)
                print(f"[{sub}] Progress saved. Exiting.")
                continue

        else:
            raise RuntimeError(f"Unknown task {args.task}.")

        print(f"[{sub}] Done with task={args.task}.\n")


if __name__ == "__main__":
    main()
