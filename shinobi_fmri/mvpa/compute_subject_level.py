import nilearn
import os
import os.path as op
import pandas as pd
import numpy as np
import shinobi_behav
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.plotting import plot_stat_map, show
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-06",
    type=str,
    help="Subject to process",
)

def main():

    path_to_data = shinobi_behav.DATA_PATH
    model = "full"
    CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT']#, 'Kill', 'HealthLoss']#'HealthGain', 'UP']
    #additional_contrasts = ['HIT+JUMP-RIGHT-LEFT-UP-DOWN', 'RIGHT+LEFT+UP+DOWN-HIT-JUMP']
    contrasts = CONDS_LIST# + additional_contrasts
    sub = args.subject
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

    z_maps = []
    contrast_label = []
    session_label = []
    for contrast in contrasts:
        z_maps_fpath = op.join(path_to_data, "processed", "z_maps", "ses-level", contrast)
        for z_map_fname in os.listdir(z_maps_fpath):
            if model in z_map_fname:
                if sub in z_map_fname:
                    session = z_map_fname.split("_")[1]
                    z_maps.append(op.join(z_maps_fpath, z_map_fname))
                    contrast_label.append(contrast)
                    session_label.append(session)


    decoder = Decoder(estimator='svc', mask=mask_fname, standardize=False,
                    screening_percentile=5, cv=LeaveOneGroupOut(), n_jobs=-1, verbose=1)
    decoder.fit(z_maps, contrast_label, groups=session_label)

    classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
    chance_level = 1. / len(np.unique(contrast_label))
    print('Classification accuracy: {:.4f} / Chance level: {}'.format(
        classification_accuracy, chance_level))
    
    for cond in contrasts:
        weight_img = decoder.coef_img_[cond]
        plot_stat_map(weight_img, bg_img=anat_fname, title=f"SVM weights {cond}")

if __name__ == "__main__":
    main()