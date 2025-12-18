import warnings
warnings.filterwarnings("ignore")
import os
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
import logging

def create_common_masker(path_to_data, subjects, masker_kwargs=None, logger=None):
    """
    Create a common NiftiMasker by resampling subject-specific brain masks 
    to a common space.
    """
    if masker_kwargs is None:
        masker_kwargs = {}

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

    masker = NiftiMasker(**masker_kwargs)
    masker.fit(mask_files)
    return masker, target_affine, target_shape


def load_zmaps_for_subject(sub, model, contrasts, path_to_data, target_affine, target_shape, logger=None):
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

    # 1) Load from main z_maps folder
    for contrast in contrasts:
        z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
        if not op.exists(z_maps_fpath):
            continue

        file_list = os.listdir(z_maps_fpath)
        for z_map_fname in tqdm(file_list, desc=f"Shinobi z_maps ({sub} {contrast})", leave=False):
            if model in z_map_fname and sub in z_map_fname:
                try:
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


def main(args, logger=None):

    np.random.seed(42)  # Global base seed for reproducibility
    path_to_data = config.DATA_PATH
    model = "simple"
    CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'Kill', 'HealthLoss']
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = config.SUBJECTS
    screening_percentile = args.screening
    n_jobs = args.n_jobs

    # Build a common masker
    all_subjects = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
    masker, target_affine, target_shape = create_common_masker(path_to_data, all_subjects, logger=logger)

    for sub in subjects:
        try:
            mvpa_results_path = op.join(path_to_data, "processed", f"mvpa_results_with_hcpmin_s{screening_percentile}")
            os.makedirs(mvpa_results_path, exist_ok=True)
            decoder_fname = f"{sub}_{model}_decoder.pkl"
            decoder_pkl_path = op.join(mvpa_results_path, decoder_fname)

            if logger:
                logger.log_computation_start(f"MVPA_{sub}", decoder_pkl_path)

            results_dict = {}

            # Load subject's z-maps
            z_maps, contrast_label, session_label = load_zmaps_for_subject(
                sub, model, CONDS_LIST, path_to_data, target_affine, target_shape, logger=logger
            )
            
            if not z_maps:
                if logger:
                    logger.warning(f"No z-maps for {sub}, skipping")
                continue

            class_list = sorted(np.unique(contrast_label))

            # --------------------------------------------------------------------
            # CLASSIF
            # --------------------------------------------------------------------
            if logger:
                logger.info("Computing Decoder...")
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
            decoder.fit(z_maps, contrast_label, groups=session_label)

            # Compute confusion matrices and accuracies
            y_pred = cross_val_predict(decoder, z_maps, contrast_label, groups=session_label, cv=decoder.cv, fit_params={'groups': session_label})
            cm = confusion_matrix(contrast_label, y_pred, labels=class_list)

            # --------------------------------------------------------------------
            # DUMMY CLASSIF
            # --------------------------------------------------------------------
            if logger:
                logger.info("Computing Dummy Decoder...")
            dummy_decoder = Decoder(
                estimator=DummyClassifier(strategy='prior'),
                mask=masker,
                standardize=True,
                scoring='balanced_accuracy',
                screening_percentile=screening_percentile,
                cv=LeaveOneGroupOut(),
                n_jobs=n_jobs,
                verbose=1
            )
            dummy_decoder.fit(z_maps, contrast_label, groups=session_label)
            from sklearn.model_selection import cross_val_score
            # Compute confusion matrices and accuracies
            scores = cross_val_score(decoder, z_maps, contrast_label, groups=session_label, scoring='balanced_accuracy', cv=decoder.cv, fit_params={'groups': session_label})
            dummy_scores = cross_val_score(dummy_decoder, z_maps, contrast_label, groups=session_label, scoring='balanced_accuracy', cv=decoder.cv, fit_params={'groups': session_label})
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
                'confusion_matrices': cm,
                'balanced_accuracy': scores,
                't_stat': t_stat,
                'p_val': p_val,
            })

            with open(decoder_pkl_path, 'wb') as f:
                pickle.dump(results_dict, f)
            
            if logger:
                logger.log_computation_success(f"MVPA_results_{sub}", decoder_pkl_path)
            
            # Save weight maps to disk
            for class_lbl in decoder.classes_:
                w_img = decoder.coef_img_[class_lbl]
                out_fname = f"{sub}_{class_lbl}_{model}_weights.nii.gz"
                out_path = op.join(mvpa_results_path, 'weight_maps', out_fname)
                os.makedirs(op.dirname(out_path), exist_ok=True)
                nib.save(w_img, out_path)
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
    parser.add_argument("-nj", "--n_jobs", default=-1, type=int,
                        help="Number of jobs for parallel processing (default: -1).")
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
