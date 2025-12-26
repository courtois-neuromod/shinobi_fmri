import warnings
import os
import sys

# Suppress all warnings including sklearn confusion matrix warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
import os.path as op
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import shinobi_fmri.config as config
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.decoding import Decoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from scipy import stats
from shinobi_fmri.utils.logger import ShinobiLogger
from shinobi_fmri.utils.provenance import create_metadata, save_sidecar_metadata, create_dataset_description
import logging

def create_common_masker(path_to_data, subjects, logger=None):
    """
    Create a common NiftiMasker by resampling subject-specific brain masks
    to a common space.
    """
    if logger:
        logger.info("Creating a common NiftiMasker. Resampling subject-specific brain masks...")
    else:
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
        try:
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
        except Exception as e:
            if logger:
                logger.error(f"Error creating mask for {sub}: {e}")
            else:
                print(f"Error creating mask for {sub}: {e}")
            continue

    masker = NiftiMasker()
    masker.fit(mask_files)
    return masker, target_affine, target_shape


def load_zmaps_for_subject(sub, contrasts, path_to_data, target_affine, target_shape, logger=None):
    """
    Load z-maps for a given subject from both the main z_maps folder
    and the hcp_results folder.
    Returns:
        z_maps (list of Nifti images)
        contrast_label (list of str)
        session_label (list of str)
    """
    msg = f"[{sub}] Loading z-maps from local and HCP-based results..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    z_maps = []
    contrast_label = []
    session_label = []

    # 1) Load from main z_maps folder (new structure: processed/session-level/sub-XX/ses-YY/z_maps/)
    # We need to search through the session-level directory for this subject
    session_level_dir = op.join(path_to_data, "processed", "session-level", sub)

    if op.exists(session_level_dir):
        # Iterate through all sessions for this subject
        for ses_dir in os.listdir(session_level_dir):
            ses_path = op.join(session_level_dir, ses_dir)
            if not op.isdir(ses_path):
                continue

            z_maps_dir = op.join(ses_path, "z_maps")
            if not op.exists(z_maps_dir):
                continue

            # Load z-maps for the requested contrasts
            for z_map_fname in tqdm(os.listdir(z_maps_dir), desc=f"Shinobi z_maps ({sub} {ses_dir})", leave=False):
                if not z_map_fname.endswith("stat-z.nii.gz"):
                    continue

                # Parse the contrast from the filename
                # Format: sub-XX_ses-YY_task-shinobi_contrast-CONDITION_stat-z.nii.gz
                try:
                    # Extract contrast name from filename
                    contrast_part = [p for p in z_map_fname.split("_") if "contrast-" in p][0]
                    contrast = contrast_part.replace("contrast-", "")

                    if contrast not in contrasts:
                        continue

                    niimap = image.load_img(op.join(z_maps_dir, z_map_fname))
                    # Resample the image to the same shape+affine for your common mask
                    resampled_img = image.resample_img(
                        niimap,
                        target_affine=target_affine,
                        target_shape=target_shape
                    )
                    z_maps.append(resampled_img)
                    contrast_label.append(contrast)
                    session_label.append(ses_dir)
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to load/resample {z_map_fname}: {e}")

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
                    try:
                        niimap = image.load_img(fpath)
                        resampled_img = image.resample_img(
                            niimap,
                            target_affine=target_affine,
                            target_shape=target_shape
                        )
                        z_maps.append(resampled_img)
                        session_label.append('_'.join(runfolder.split('_')[2:]))
                        contrast_label.append(cond)
                    except Exception as e:
                         if logger:
                            logger.warning(f"Failed to load/resample {fpath}: {e}")

    msg = f"[{sub}] Loaded {len(z_maps)} z-maps total."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return z_maps, contrast_label, session_label


def run_permutations(sub, z_maps, contrast_label, session_label, masker, screening_percentile, n_jobs,
                      perm_start, perm_end, n_permutations, output_path, logger=None):
    """
    Run permutation testing for MVPA.
    """
    # Set base seed
    np.random.seed(42)

    # Generate all permutation seeds upfront for reproducibility
    perm_seeds = [42 + i for i in range(n_permutations)]

    if perm_end is None:
        perm_end = n_permutations

    perm_results = []

    for perm_idx in range(perm_start, perm_end):
        if logger:
            logger.info(f"[{sub}] Running permutation {perm_idx + 1}/{n_permutations}")
        else:
            print(f"[{sub}] Running permutation {perm_idx + 1}/{n_permutations}")

        # Set seed for this permutation
        np.random.seed(perm_seeds[perm_idx])

        # Shuffle labels
        permuted_labels = np.random.permutation(contrast_label)

        # Create decoder
        decoder_perm = Decoder(
            estimator=LinearSVC(random_state=42),
            mask=masker,
            standardize=True,
            scoring='balanced_accuracy',
            screening_percentile=screening_percentile,
            cv=LeaveOneGroupOut(),
            n_jobs=n_jobs,
            verbose=0
        )

        # Fit with permuted labels
        decoder_perm.fit(z_maps, permuted_labels, groups=session_label)

        # Extract per-class scores
        scores_per_class = {}
        for class_label in decoder_perm.cv_scores_:
            scores_per_class[class_label] = np.array(decoder_perm.cv_scores_[class_label])

        # Store result
        perm_results.append({
            'perm_index': perm_idx,
            'scores_per_class': scores_per_class
        })

        # Save progress every 10 permutations
        if (perm_idx - perm_start + 1) % 10 == 0:
            with open(output_path, 'wb') as f:
                pickle.dump(perm_results, f)
            if logger:
                logger.info(f"[{sub}] Saved progress: {perm_idx - perm_start + 1} permutations")

    # Final save
    with open(output_path, 'wb') as f:
        pickle.dump(perm_results, f)

    if logger:
        logger.info(f"[{sub}] Completed permutations {perm_start}-{perm_end-1}")

    return perm_results


def main(args, logger=None):

    np.random.seed(42)  # Global base seed for reproducibility
    path_to_data = config.DATA_PATH
    CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'Kill', 'HealthLoss']
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = config.SUBJECTS
    screening_percentile = args.screening
    n_jobs = args.n_jobs
    n_permutations = args.n_permutations
    perm_start = args.perm_start
    perm_end = args.perm_end if args.perm_end is not None else n_permutations

    # Build a common masker
    all_subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
    masker, target_affine, target_shape = create_common_masker(path_to_data, all_subjects, logger=logger)

    for sub in subjects:
        try:
            mvpa_results_path = op.join(path_to_data, "processed", f"mvpa_results_s{screening_percentile}")
            os.makedirs(mvpa_results_path, exist_ok=True)
            decoder_fname = f"{sub}_decoder.pkl"
            decoder_pkl_path = op.join(mvpa_results_path, decoder_fname)

            if logger:
                logger.log_computation_start(f"MVPA_{sub}", decoder_pkl_path)

            results_dict = {}

            # Load subject's z-maps
            z_maps, contrast_label, session_label = load_zmaps_for_subject(
                sub, CONDS_LIST, path_to_data, target_affine, target_shape, logger=logger
            )

            if not z_maps:
                if logger:
                    logger.warning(f"No z-maps for {sub}, skipping")
                continue

            class_list = sorted(np.unique(contrast_label))

            # --------------------------------------------------------------------
            # PERMUTATION TESTING
            # --------------------------------------------------------------------
            if n_permutations > 0 and perm_start < perm_end:
                # Running permutations only
                perm_output_dir = op.join(mvpa_results_path, f"{sub}_permutations")
                os.makedirs(perm_output_dir, exist_ok=True)
                perm_output_path = op.join(perm_output_dir, f"perm_{perm_start}_{perm_end-1}.pkl")

                if logger:
                    logger.log_computation_start(f"MVPA_perm_{sub}_{perm_start}_{perm_end-1}", perm_output_path)

                run_permutations(sub, z_maps, contrast_label, session_label, masker, screening_percentile,
                                n_jobs, perm_start, perm_end, n_permutations, perm_output_path, logger=logger)

                if logger:
                    logger.log_computation_success(f"MVPA_perm_{sub}_{perm_start}_{perm_end-1}", perm_output_path)
                continue  # Skip actual analysis when running permutations

            # --------------------------------------------------------------------
            # CLASSIF
            # --------------------------------------------------------------------
            if logger:
                logger.info("Computing Decoder...")

            # Create decoder with CV - fit() will perform CV internally
            decoder = Decoder(
                estimator=LinearSVC(random_state=42),
                mask=masker,
                standardize=True,
                scoring='balanced_accuracy',
                screening_percentile=screening_percentile,
                cv=LeaveOneGroupOut(),
                n_jobs=n_jobs,
                verbose=1
            )

            # Fit decoder with groups - CV happens internally
            decoder.fit(z_maps, contrast_label, groups=session_label)

            # Extract CV scores per class from decoder
            scores_per_class = {}
            for class_label in decoder.cv_scores_:
                scores_per_class[class_label] = np.array(decoder.cv_scores_[class_label])

            # Compute mean balanced accuracy across all folds
            all_fold_scores = []
            for class_scores in decoder.cv_scores_.values():
                all_fold_scores.extend(class_scores)
            scores = np.array(all_fold_scores)

            # Compute confusion matrices for each CV fold
            if logger:
                logger.info("Computing confusion matrices...")
            fold_confusions = []
            cv = LeaveOneGroupOut()
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(z_maps, contrast_label, groups=session_label)):
                # Create a new decoder for this fold
                fold_decoder = Decoder(
                    estimator=LinearSVC(random_state=42),
                    mask=masker,
                    standardize=True,
                    screening_percentile=screening_percentile,
                    n_jobs=1,  # Single job for fold
                    verbose=0
                )

                # Get train/test data
                train_maps = [z_maps[i] for i in train_idx]
                test_maps = [z_maps[i] for i in test_idx]
                train_labels = [contrast_label[i] for i in train_idx]
                test_labels = [contrast_label[i] for i in test_idx]

                # Fit and predict
                fold_decoder.fit(train_maps, train_labels)
                y_pred = fold_decoder.predict(test_maps)

                # Compute confusion matrix
                cm_fold = confusion_matrix(test_labels, y_pred, labels=class_list)
                fold_confusions.append(cm_fold)

            fold_confusions = np.array(fold_confusions)

            # --------------------------------------------------------------------
            # DUMMY CLASSIF
            # --------------------------------------------------------------------
            if logger:
                logger.info("Computing Dummy Decoder...")
            dummy_decoder = Decoder(
                estimator=DummyClassifier(strategy='prior', random_state=42),
                mask=masker,
                standardize=True,
                scoring='balanced_accuracy',
                screening_percentile=screening_percentile,
                cv=LeaveOneGroupOut(),
                n_jobs=n_jobs,
                verbose=1
            )

            # Fit dummy decoder with groups
            dummy_decoder.fit(z_maps, contrast_label, groups=session_label)

            # Extract dummy scores
            dummy_all_fold_scores = []
            for class_scores in dummy_decoder.cv_scores_.values():
                dummy_all_fold_scores.extend(class_scores)
            dummy_scores = np.array(dummy_all_fold_scores)

            t_stat, p_val = stats.ttest_ind(scores, dummy_scores)
            
            msg = f"[{sub}] T-test between decoder and dummy classifier: t-stat={t_stat:.3f}, p-value={p_val:.3f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            # --------------------------------------------------------------------
            # SAVE RESULTS
            # --------------------------------------------------------------------
            # Save or update results_dict
            results_dict.update({
                'decoder': decoder,
                'contrast_label': contrast_label,
                'class_labels': class_list,
                'scores_per_class': scores_per_class,
                'balanced_accuracy_scores': scores,
                'mean_balanced_accuracy': np.mean(scores),
                'confusion_matrices': {'fold_confusions': fold_confusions},  # NB6 format
                't_stat': t_stat,
                'p_val': p_val,
            })

            with open(decoder_pkl_path, 'wb') as f:
                pickle.dump(results_dict, f)

            # Save provenance metadata
            metadata = create_metadata(
                description=f"MVPA decoder results for {sub}",
                script_path=__file__,
                output_files=[decoder_pkl_path],
                parameters={
                    'estimator': 'LinearSVC',
                    'random_state': 42,
                    'screening_percentile': screening_percentile,
                    'scoring': 'balanced_accuracy',
                    'cv': 'LeaveOneGroupOut',
                    'n_jobs': n_jobs,
                    'standardize': True,
                    'conditions': CONDS_LIST,
                    'n_classes': len(class_list),
                    'n_samples': len(z_maps),
                },
                subject=sub,
                session=None,  # Multi-session MVPA
                additional_info={
                    'analysis_type': 'MVPA',
                    'classes': class_list,
                    'mean_balanced_accuracy': float(np.mean(scores)),
                    't_stat': float(t_stat),
                    'p_val': float(p_val),
                }
            )
            # Save as .json next to .pkl file
            metadata_path = decoder_pkl_path.replace('.pkl', '.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            if logger:
                logger.debug(f"Saved metadata: {metadata_path}")

            if logger:
                logger.log_computation_success(f"MVPA_results_{sub}", decoder_pkl_path)
            
            # Save weight maps to disk
            for class_lbl in decoder.classes_:
                w_img = decoder.coef_img_[class_lbl]
                out_fname = f"{sub}_{class_lbl}_weights.nii.gz"
                out_path = op.join(mvpa_results_path, 'weight_maps', out_fname)
                os.makedirs(op.dirname(out_path), exist_ok=True)
                nib.save(w_img, out_path)

                # Save provenance metadata for weight map
                weight_metadata = create_metadata(
                    description=f"MVPA weight map for class {class_lbl}",
                    script_path=__file__,
                    output_files=[out_path],
                    parameters={
                        'class': class_lbl,
                        'estimator': 'LinearSVC',
                        'screening_percentile': screening_percentile,
                    },
                    subject=sub,
                    session=None,
                    additional_info={
                        'analysis_type': 'MVPA_weights',
                        'decoder_results': op.basename(decoder_pkl_path),
                    }
                )
                save_sidecar_metadata(out_path, weight_metadata, logger=logger)

                if logger:
                    logger.debug(f"Saved weight map: {out_path}")

            if logger:
                logger.info(f"[{sub}] Finished task.\n")
                
        except Exception as e:
            if logger:
                logger.log_computation_error(f"MVPA_{sub}", e)
            else:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", default=None, type=str,
                        help="Subject to process (e.g. 'sub-01').")
    parser.add_argument("--screening", default=20, type=int,
                        help="Percentile for screening (default: 20).")
    parser.add_argument("-nj", "--n-jobs", default=-1, type=int, dest='n_jobs',
                        help="Number of jobs for parallel processing (default: -1).")
    parser.add_argument("--n-permutations", default=0, type=int,
                        help="Total number of permutations (default: 0 = no permutation testing).")
    parser.add_argument("--perm-start", default=0, type=int,
                        help="Starting permutation index for this job (default: 0).")
    parser.add_argument("--perm-end", default=None, type=int,
                        help="Ending permutation index for this job (default: None = all).")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files",
    )
    args = parser.parse_args()
    
    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="MVPA",
        subject=args.subject,
        log_dir=args.log_dir,
        verbosity=log_level
    )
    
    try:
        main(args, logger=logger)
    finally:
        logger.close()
