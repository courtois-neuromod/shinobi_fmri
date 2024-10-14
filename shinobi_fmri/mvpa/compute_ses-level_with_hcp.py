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
import os.path as op
import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default=None,
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

#def main():
path_to_data = shinobi_behav.DATA_PATH
models = ["simple"]
model = "simple"
CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'Kill', 'HealthLoss']#, 'Kill', 'HealthLoss']#'HealthGain', 'UP']
#additional_contrasts = ['HIT+JUMP-RIGHT-LEFT-UP-DOWN', 'RIGHT+LEFT+UP+DOWN-HIT-JUMP']
contrasts = CONDS_LIST# + additional_contrasts
if args.subject is not None:
    subjects = [args.subject]
else:
    subjects = shinobi_behav.SUBJECTS



def create_common_masker(path_to_data, subjects, masker_kwargs=None):
    """
    Create a common NiftiMasker by resampling subject-specific brain masks to a common space.

    Parameters
    ----------
    path_to_data : str
        The base directory where the data is stored.
    subjects : list of str
        A list of subject identifiers (e.g., ['sub-01', 'sub-02']).
    masker_kwargs : dict, optional
        Additional keyword arguments to pass to the NiftiMasker.

    Returns
    -------
    masker : NiftiMasker
        A fitted NiftiMasker object using the resampled masks.
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

    if op.isfile(op.join(mvpa_results_path, decoder_fname)):
        with open(op.join(mvpa_results_path, decoder_fname), 'rb') as f:
            decoder = pickle.load(f)
    else:

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
        #for model in models:
        z_maps = []
        contrast_label = []
        session_label = []

        for contrast in contrasts:
            z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
            for z_map_fname in os.listdir(z_maps_fpath):
                if model in z_map_fname:
                    if sub in z_map_fname:
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
                fpath = op.join(subfolder, runfolder, 'z_score_maps', '{}.nii.gz'.format(cond))
                print(f'Loading {fpath}')
                niimap = image.load_img(fpath)
                resampled_img = image.resample_img(niimap, target_affine=target_affine, target_shape=target_shape)
                z_maps.append(resampled_img)
                session_label.append('_'.join(runfolder.split('_')[2:]))
                contrast_label.append(cond)

        decoder = Decoder(estimator='svc', mask=masker, standardize=False, scoring='accuracy',
                        screening_percentile=5, cv=LeaveOneGroupOut(), n_jobs=-1, verbose=1)
        decoder.fit(z_maps, contrast_label, groups=session_label)

        # Save decoder
        with open(op.join(mvpa_results_path, f"{sub}_{model}_decoder.pkl"), 'wb') as f:
            pickle.dump(decoder, f)

        classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
        chance_level = 1. / len(np.unique(contrast_label))
        print(f'Decoding : {sub} {model}')
        print('Classification accuracy: {:.4f} / Chance level: {}'.format(
            classification_accuracy, chance_level))
        
    # Plot weights
    for cond in np.unique(contrast_label):
        output_fname = op.join("./", "reports", "figures", "ses-level", cond, "MVPA", f"{sub}_{cond}_{model}_mvpa.png")
        os.makedirs(op.join("./", "reports", "figures", "ses-level", cond, "MVPA"), exist_ok=True)
        weight_img = decoder.coef_img_[cond]
        plot_stat_map(weight_img, bg_img=anat_fname, title=f"SVM weights {cond}", output_file=output_fname)
        nib.save(weight_img, op.join(mvpa_results_path, f"{sub}_{cond}_{model}_mvpa.nii.gz"))

    # Generate confusion matrices across folds
    confusion_matrices = []
    for train, test in decoder.cv.split(z_maps, contrast_label, groups=session_label):
        decoder.fit(np.array(z_maps)[train], np.array(contrast_label)[train], groups=np.array(session_label)[train])
        y_pred = decoder.predict(np.array(z_maps)[test])
        y_true = np.array(contrast_label)[test]
        # Each row is normalized by the sum of the elements in that row (i.e., the total number of actual instances for that class).
        confusion_mat = confusion_matrix(y_true, y_pred, normalize='true', labels=decoder.classes_) 
        confusion_matrices.append(confusion_mat)

    # Plot confusion matrices
    averaged_confusion_matrix = np.mean(confusion_matrices, axis=0)
    std_confusion_matrix = np.std(confusion_matrices, axis=0)
    plt.figure(figsize=(10, 10))
    sbn.heatmap(averaged_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=decoder.classes_, yticklabels=decoder.classes_)
    output_fname = op.join("./", "reports", "figures", "ses-level", "confusion_matrices", f"{sub}_{model}_averaged_confusion_matrix.png")
    os.makedirs(op.join("./", "reports", "figures", "ses-level", "confusion_matrices"), exist_ok=True)
    plt.savefig(output_fname)
    print(f'Saving {output_fname}')
    plt.close()
    plt.figure(figsize=(10, 10))
    sbn.heatmap(std_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=decoder.classes_, yticklabels=decoder.classes_)
    output_fname = op.join("./", "reports", "figures", "ses-level", "confusion_matrices", f"{sub}_{model}_std_confusion_matrix.png")
    plt.savefig(output_fname)
    print(f'Saving {output_fname}')
    plt.close()