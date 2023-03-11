
import nilearn
import os
import os.path as op
import pandas as pd
import numpy as np
import shinobi_behav

from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

path_to_data = shinobi_behav.DATA_PATH
model = "full"
CONDS_LIST = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'UP', 'Kill', 'HealthGain', 'HealthLoss']
additional_contrasts = ['HIT+JUMP-RIGHT-LEFT-UP-DOWN', 'RIGHT+LEFT+UP+DOWN-HIT-JUMP']
contrasts = CONDS_LIST + additional_contrasts
sub = "sub-01"
mask_fname = op.join(
    path_to_data,
    "cneuromod.processed",
    "smriprep",
    sub,
    "anat",
    f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
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
