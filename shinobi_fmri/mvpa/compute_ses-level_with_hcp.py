import warnings

warnings.filterwarnings("ignore")
import os
import os.path as op
import numpy as np
import pickle
import argparse
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
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

from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

##############################################################################
# ARGUMENT PARSING
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default=None,
    type=str,
    help="Subject to process (e.g. 'sub-01').",
)
parser.add_argument(
    "--task",
    choices=["classif", "perm"],
    default="classif",
    help="Task to run: 'classif' for real classification, 'perm' for permutations.",
)
parser.add_argument(
    "--perm-index",
    type=int,
    default=None,
    help="Starting permutation index. If not given, we start from the lowest missing index.",
)
args = parser.parse_args()

##############################################################################
# GLOBAL CONFIG
##############################################################################
np.random.seed(42)  # Global base seed for reproducibility
path_to_data = shinobi_behav.DATA_PATH
model = "simple"
CONDS_LIST = ["HIT", "JUMP", "DOWN", "LEFT", "RIGHT", "Kill", "HealthLoss"]
if args.subject is not None:
    subjects = [args.subject]
else:
    subjects = shinobi_behav.SUBJECTS

screening_percentile = 20
n_permutations = 10
n_jobs = 8


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
            mask_fname, target_affine=target_affine, target_shape=target_shape
        )
        mask_files.append(mask_resampled)

    masker = NiftiMasker(**masker_kwargs)
    masker.fit(mask_files)
    return masker, target_affine, target_shape


def load_zmaps_for_subject(
    sub, model, contrasts, path_to_data, target_affine, target_shape
):
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
        z_maps_fpath = op.join(
            path_to_data, "processed", "z_maps", "ses-level", contrast
        )
        if not op.exists(z_maps_fpath):
            continue

        file_list = os.listdir(z_maps_fpath)
        for z_map_fname in tqdm(
            file_list, desc=f"Shinobi z_maps ({sub} {contrast})", leave=False
        ):
            if model in z_map_fname and sub in z_map_fname:
                session = z_map_fname.split("_")[1]
                niimap = image.load_img(op.join(z_maps_fpath, z_map_fname))
                # Resample the image to the same shape+affine for your common mask
                resampled_img = image.resample_img(
                    niimap, target_affine=target_affine, target_shape=target_shape
                )
                z_maps.append(resampled_img)
                contrast_label.append(contrast)
                session_label.append(session)

    # 2) Load from HCP-based results
    subfolder = op.join(path_to_data, "hcp_results", sub)
    if op.exists(subfolder):
        runfolders = [f for f in os.listdir(subfolder) if "run-" in f]
        for runfolder in tqdm(runfolders, desc=f"HCP z_maps ({sub})", leave=False):
            path_z = op.join(subfolder, runfolder, "z_score_maps")
            if not op.exists(path_z):
                continue
            conditions = [
                f.split(".")[0] for f in os.listdir(path_z) if f.endswith(".nii.gz")
            ]
            for cond in conditions:
                fpath = op.join(path_z, f"{cond}.nii.gz")
                if op.exists(fpath):
                    niimap = image.load_img(fpath)
                    resampled_img = image.resample_img(
                        niimap, target_affine=target_affine, target_shape=target_shape
                    )
                    z_maps.append(resampled_img)
                    session_label.append("_".join(runfolder.split("_")[2:]))
                    contrast_label.append(cond)

    print(f"[{sub}] Loaded {len(z_maps)} z-maps total.")
    return z_maps, contrast_label, session_label


def compute_crossval_confusions_and_accuracies(
    X_data, y_data, group_labels, estimator, cv=LeaveOneGroupOut(), n_jobs=1
):
    """
    Manually implement cross-validation to compute:
      - A confusion matrix for each fold
      - Per-label accuracies across folds

    Parameters:
      X_data       : array-like feature data
      y_data       : array-like labels
      group_labels : array-like group assignments (for CV)
      estimator    : Either an estimator class (e.g., LinearSVC) or an instance
                     (e.g., a configured Nilearn Decoder that includes a cv attribute)
      n_jobs       : Number of parallel jobs

    Returns:
      fold_confusions         : list of confusion matrices (one per fold)
      actual_per_class_accuracy: dict mapping each label to its average accuracy across folds
    """

    splits = list(cv.split(X_data, y_data, groups=group_labels))
    all_labels = np.unique(y_data)

    def _fit_and_predict_fold(train_idx, test_idx):
        clf = clone(estimator)
        import os
        import psutil

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss  # in bytes
        print(f"Memory usage: {memory_usage / (1024 ** 2):.2f} MB")

        clf.fit(X_data[train_idx], y_data[train_idx])

        y_pred = clf.predict(X_data[test_idx])
        cm = confusion_matrix(y_data[test_idx], y_pred, labels=all_labels)

        fold_acc_dict = {}
        for lbl in all_labels:
            lbl_mask = y_data[test_idx] == lbl
            if np.sum(lbl_mask) > 0:
                fold_acc_dict[lbl] = np.mean(y_pred[lbl_mask] == lbl)
        return cm, fold_acc_dict

    with parallel_backend("threading"):
        with tqdm(total=len(splits), desc="Manual CV folds") as pbar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_and_predict_fold)(train_idx, test_idx)
                for (train_idx, test_idx) in splits
            )
            pbar.update(len(splits))

    fold_confusions = []
    per_class_accuracies = {lbl: [] for lbl in all_labels}
    for cm, fold_acc_dict in results:
        fold_confusions.append(cm)
        for lbl, acc in fold_acc_dict.items():
            per_class_accuracies[lbl].append(acc)

    actual_per_class_accuracies = {
        lbl: np.mean(per_class_accuracies[lbl]) for lbl in all_labels
    }
    return fold_confusions, actual_per_class_accuracies


##############################################################################
# MAIN SCRIPT
##############################################################################
def main():
    # 1) Build or reuse a common masker
    all_subjects = ["sub-01", "sub-02", "sub-04", "sub-06"]
    masker, target_affine, target_shape = create_common_masker(
        path_to_data, all_subjects
    )

    for sub in subjects:
        print(f"\n===== Processing {sub} ===== (Task: {args.task})")
        mvpa_results_path = op.join(path_to_data, "processed", "mvpa_results_with_hcp")
        os.makedirs(mvpa_results_path, exist_ok=True)
        decoder_fname = f"{sub}_{model}_decoder.pkl"
        decoder_pkl_path = op.join(mvpa_results_path, decoder_fname)

        # Load existing main results dictionary if available
        if op.isfile(decoder_pkl_path):
            with open(decoder_pkl_path, "rb") as f:
                results_dict = pickle.load(f)
            print(f"[{sub}] Loaded existing results from {decoder_pkl_path}.")
        else:
            print(f"[{sub}] No existing results found, starting fresh.")
            results_dict = {}

        # Try extracting relevant fields from results_dict if present
        contrast_label_stored = results_dict.get("contrast_label", None)
        confusion_matrices_dict = results_dict.get("confusion_matrices_dict", {})
        actual_per_class_accuracies = results_dict.get(
            "actual_per_class_accuracies", None
        )
        completed_permutations = results_dict.get("completed_permutations", 0)
        class_list = results_dict.get("class_labels", None)
        p_values = results_dict.get(
            "p_values", None
        )  # might exist if previously computed
        # Load subject's z-maps
        z_maps, contrast_label_loaded, session_label = load_zmaps_for_subject(
            sub, model, CONDS_LIST, path_to_data, target_affine, target_shape
        )
        # Precompute features from the z-maps (this is the solution offered in 2)
        print(f"[{sub}] Transforming z-maps using masker...")
        # X_features = masker.transform(z_maps)
        if contrast_label_stored:
            # If we already have a label stored, keep using it (consistency)
            # Otherwise, rely on what's newly loaded
            contrast_label = contrast_label_stored
        else:
            contrast_label = contrast_label_loaded

        # We'll store class_labels once we know them from the data
        if not class_list and len(contrast_label) > 0:
            class_list = sorted(np.unique(contrast_label))

        # --------------------------------------------------------------------
        # 1) CLASSIF TASK
        # --------------------------------------------------------------------
        if args.task == "classif":
            print(
                f"[{sub}] Running manual cross-validation to get fold-wise confusion..."
            )

            # If there's no real data
            if len(contrast_label) == 0:
                print(f"[{sub}] No data found, skipping classification.")
                continue

            decoder_main = Decoder(
                estimator=LinearSVC(random_state=42),
                mask=masker,
                standardize=True,
                scoring="balanced_accuracy",
                screening_percentile=screening_percentile,
                cv=None,
                n_jobs=1,
                verbose=0,
            )

            fold_confusions, actual_per_class_accuracies = (
                compute_crossval_confusions_and_accuracies(
                    np.array(z_maps),
                    np.array(contrast_label),
                    np.array(session_label),
                    estimator=decoder_main,
                    n_jobs=n_jobs,
                )
            )

            confusion_matrices_dict = {"fold_confusions": fold_confusions}

            print("[Per-class accuracy across folds]:")
            for lbl in sorted(set(contrast_label)):
                acc_val = actual_per_class_accuracies.get(lbl, np.nan)
                print(f"  {lbl}: {acc_val:.4f}")

            # Fit a final Decoder on all data to get weight maps
            print(
                f"[{sub}] Training final Decoder on entire dataset to get weight maps..."
            )
            final_decoder = Decoder(
                estimator=LinearSVC(random_state=42),
                mask=masker,
                standardize=True,
                scoring="balanced_accuracy",
                screening_percentile=screening_percentile,
                cv=LeaveOneGroupOut(),
                n_jobs=n_jobs,
                verbose=0,
            )
            final_decoder.fit(z_maps, contrast_label, groups=session_label)

            # Save or update results_dict
            results_dict.update(
                {
                    "decoder": final_decoder,
                    "contrast_label": contrast_label,
                    "class_labels": class_list,
                    "confusion_matrices_dict": confusion_matrices_dict,
                    "actual_per_class_accuracies": actual_per_class_accuracies,
                    "completed_permutations": completed_permutations,
                    "p_values": p_values,  # might exist if computed earlier, keep it
                }
            )

            with open(decoder_pkl_path, "wb") as f:
                pickle.dump(results_dict, f)
            print(f"[{sub}] Classification results saved to {decoder_pkl_path}.")

            # Save weight maps to disk
            for class_lbl in final_decoder.classes_:
                w_img = final_decoder.coef_img_[class_lbl]
                out_fname = f"{sub}_{class_lbl}_{model}_weights.nii.gz"
                out_path = op.join(mvpa_results_path, "weight_maps", out_fname)
                os.makedirs(op.dirname(out_path), exist_ok=True)
                nib.save(w_img, out_path)

        # --------------------------------------------------------------------
        # 2) PERM TASK
        # --------------------------------------------------------------------
        elif args.task == "perm":
            # This folder will hold individual permutation results
            perm_folder = op.join(mvpa_results_path, f"{sub}_{model}_permutations")
            os.makedirs(perm_folder, exist_ok=True)

            # Figure out how many permutations are already done
            existing_perms = [
                f
                for f in os.listdir(perm_folder)
                if f.startswith("perm_") and f.endswith(".pkl")
            ]
            done_indices = set()
            for fname in existing_perms:
                # e.g. 'perm_12.pkl' -> index=12
                idx_str = fname.split("_")[1].split(".")[0]
                done_indices.add(int(idx_str))
            completed_permutations = len(done_indices)
            results_dict["completed_permutations"] = completed_permutations

            with open(decoder_pkl_path, "wb") as f:
                pickle.dump(results_dict, f)

            # If we have data, ensure we know what classes exist
            if len(contrast_label) > 0 and not class_list:
                class_list = sorted(np.unique(contrast_label))

            if completed_permutations >= n_permutations:
                print(f"[{sub}] All permutations completed ({completed_permutations}).")
            else:
                # find the next missing index
                if args.perm_index is not None:
                    current_index = args.perm_index
                else:
                    # find the first missing
                    candidates = [
                        i for i in range(n_permutations) if i not in done_indices
                    ]
                    if len(candidates) == 0:
                        print(f"[{sub}] All permutations completed! No missing index.")
                        current_index = n_permutations
                    else:
                        current_index = candidates[0]

                print(
                    f"[{sub}] Starting permutations at index={current_index} (up to {n_permutations-1})..."
                )

                while current_index < n_permutations:
                    if current_index in done_indices:
                        print(
                            f"[{sub}] Perm {current_index} is already computed. Skipping."
                        )
                        current_index += 1
                        continue

                    if len(contrast_label) == 0:
                        print(
                            f"[{sub}] No real data to permute. Skipping permutations."
                        )
                        break

                    print(f"[{sub}] Computing permutation {current_index}...")

                    np.random.seed(42 + current_index)  # stable, reproducible
                    perm_labels = np.random.permutation(contrast_label)

                    decoder_perm = Decoder(
                        estimator=LinearSVC(random_state=42),
                        mask=masker,
                        standardize=True,
                        scoring="balanced_accuracy",
                        screening_percentile=screening_percentile,
                        cv=None,
                        n_jobs=1,
                        verbose=0,
                    )
                    try:
                        with parallel_backend("threading"):
                            with tqdm_joblib(
                                tqdm(
                                    desc=f"[{sub}] Perm {current_index} CV Progress",
                                    leave=False,
                                )
                            ):
                                fold_confusions, perm_per_class_accuracies = (
                                    compute_crossval_confusions_and_accuracies(
                                        np.array(z_maps),
                                        np.array(perm_labels),
                                        np.array(session_label),
                                        estimator=decoder_perm,
                                        n_jobs=n_jobs,
                                    )
                                )
                    except KeyboardInterrupt:
                        print(
                            f"[{sub}] Interrupted at permutation {current_index}. Saving progress..."
                        )
                        break

                    # Save the single-permutation results to a dedicated file
                    perm_results = {
                        "index": current_index,
                        "class_list": class_list,
                        "acc": perm_per_class_accuracies,
                    }
                    single_perm_path = op.join(perm_folder, f"perm_{current_index}.pkl")
                    with open(single_perm_path, "wb") as pf:
                        pickle.dump(perm_results, pf)

                    done_indices.add(current_index)
                    completed_permutations = len(done_indices)
                    results_dict["completed_permutations"] = completed_permutations

                    with open(decoder_pkl_path, "wb") as f:
                        pickle.dump(results_dict, f)

                    print(
                        f"[{sub}] => Perm {current_index} done. "
                        f"{completed_permutations}/{n_permutations} completed."
                    )

                    if completed_permutations == n_permutations:
                        break
                    current_index += 1
                    break
            # ----------------------------------------------------------------
            # If we now have all permutations, gather them into one file
            # ----------------------------------------------------------------
            if completed_permutations == n_permutations:
                print(
                    f"[{sub}] All permutations done! Checking for final aggregated file..."
                )
                all_perms_path = op.join(perm_folder, "all_permutations.pkl")
                if op.exists(all_perms_path):
                    print(
                        f"[{sub}] The final aggregated file already exists: {all_perms_path}"
                    )
                else:
                    print(
                        f"[{sub}] Aggregating all permutation results into {all_perms_path}..."
                    )
                    # We'll load each perm_*.pkl
                    if class_list:
                        n_classes = len(class_list)
                        perm_array = np.full((n_permutations, n_classes), np.nan)
                        perm_files = sorted(
                            [
                                f
                                for f in os.listdir(perm_folder)
                                if f.startswith("perm_") and f.endswith(".pkl")
                            ]
                        )
                        for fname in perm_files:
                            idx_str = fname.split("_")[1].split(".")[0]
                            idx_int = int(idx_str)
                            with open(op.join(perm_folder, fname), "rb") as pf:
                                p_data = pickle.load(pf)
                            for i_class, c_lbl in enumerate(class_list):
                                perm_array[idx_int, i_class] = p_data["acc"].get(
                                    c_lbl, np.nan
                                )
                    else:
                        # No real data => no permutations to combine
                        perm_array = None

                    # Save the big array
                    with open(all_perms_path, "wb") as pf:
                        pickle.dump(
                            {"perm_array": perm_array, "class_list": class_list}, pf
                        )

                    # Delete all the perm_*.pkl files
                    print(f"[{sub}] Deleting individual perm_* files...")
                    for fname in os.listdir(perm_folder):
                        if fname.startswith("perm_") and fname.endswith(".pkl"):
                            os.remove(op.join(perm_folder, fname))

                # If classification has been done (actual_per_class_accuracies != None),
                # compute p-values. If not, we skip for now.
                if actual_per_class_accuracies and class_list:
                    with open(all_perms_path, "rb") as pf:
                        merged_data = pickle.load(pf)
                    perm_array = merged_data["perm_array"]

                    p_values = {}
                    for i_class, c_lbl in enumerate(class_list):
                        real_acc = actual_per_class_accuracies.get(c_lbl, np.nan)
                        perms_acc = perm_array[:, i_class]
                        # Some classes might have NaNs if no data
                        valid_mask = ~np.isnan(perms_acc)
                        if not np.any(valid_mask):
                            p_val = np.nan
                        else:
                            p_val = np.mean(perms_acc[valid_mask] >= real_acc)
                        p_values[c_lbl] = p_val

                    results_dict["p_values"] = p_values
                    with open(decoder_pkl_path, "wb") as f:
                        pickle.dump(results_dict, f)

                    print("[Final p-values]:")
                    for c_lbl, val in p_values.items():
                        print(f"  {c_lbl}: p={val:.4f}")

                else:
                    print(
                        f"[{sub}] Classification not done yet (or no real data). "
                        "Skipping p-value computation."
                    )
            else:
                print(
                    f"[{sub}] Not all permutations are done => no final aggregation yet."
                )

            print(f"[{sub}] Done with permutations.\n")

        else:
            raise RuntimeError(f"Unknown task {args.task}.")

        print(f"[{sub}] Finished task={args.task}.\n")


if __name__ == "__main__":
    main()
