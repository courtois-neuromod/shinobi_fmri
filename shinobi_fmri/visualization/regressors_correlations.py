import shinobi_behav
import os.path as op
import seaborn as sbn
from shinobi_fmri.annotations.annotations import trim_events_df
import os
import pandas as pd
from nilearn.signal import clean
from nilearn.image import clean_img
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
from load_confounds import Confounds
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

figures_path = shinobi_behav.figures_path #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
path_to_data = shinobi_behav.path_to_data  #'/media/storage/neuromod/shinobi_data/'
subjects = shinobi_behav.subjects
t_r=1.49
hrf_model='spm'

regressors_dict = {'regressors':[],
            'subject':[],
            'session':[],
            'run':[]}

for sub in subjects:
    sessions = os.listdir(op.join(path_to_data, "shinobi", sub))
    for ses in sorted(sessions):
        print(f"Processing {sub} {ses}")
        runs = [
            filename[-12]
            for filename in os.listdir(
                op.join(path_to_data, "shinobi", sub, ses, "func")
            )
            if "events.tsv" in filename
        ]
        for run in sorted(runs):
            try:
                events_fname = op.join(
                    path_to_data, "processed", "annotations", f"{sub}_{ses}_run-0{run}.csv"
                )
                fmri_fname = op.join(
                    path_to_data,
                    "shinobi",
                    "derivatives",
                    "fmriprep-20.2lts",
                    "fmriprep",
                    sub,
                    ses,
                    "func",
                    f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                )
                confounds_obj = Confounds(
                    strategy=["high_pass", "motion"],
                    motion="full")
                confounds = confounds_obj.load(fmri_fname)
                run_events = pd.read_csv(events_fname)
                events_df = trim_events_df(run_events, trim_by="event")
                fmri_img = clean_img(
                    fmri_fname,
                    standardize=True,
                    detrend=True,
                    high_pass=None,
                    t_r=1.49,
                    ensure_finite=True,
                    confounds=None,
                )

                # Generate design matrix
                bold_shape = fmri_img.shape
                n_slices = bold_shape[-1]
                frame_times = np.arange(n_slices) * t_r
                design_matrix_raw = make_first_level_design_matrix(
                    frame_times,
                    events=events_df,
                    drift_model=None,
                    hrf_model=hrf_model,
                    add_regs=pd.DataFrame(confounds, columns=confounds_obj.columns_),
                    add_reg_names=None
                )
                regressors_clean = clean(
                    design_matrix_raw.to_numpy(),
                    detrend=True,
                    standardize=True,
                    high_pass=None,
                    t_r=1.49,
                    ensure_finite=True,
                    confounds=None,
                )
                design_matrix_clean = pd.DataFrame(
                    regressors_clean, columns=design_matrix_raw.columns.to_list()
                )
                regressors_dict['regressors'].append(design_matrix_clean)
                regressors_dict['subject'].append(sub)
                regressors_dict['session'].append(ses)
                regressors_dict['run'].append(run)
                print(confounds.shape)
            except Exception as e:
                print(e)
                print(f'Run {run} : events_file empty or missing')


regressors_dict_fname = op.join(
    path_to_data, "processed", 'z_maps', "run-level", "regressors_dict.pkl"
)
with open(regressors_dict_fname, "wb") as f:
    pickle.dump(regressors_dict, f)
0/0

#with open(regressors_dict_fname, "rb") as f:
#    regressors_dict = pickle.load(f)

regressors_dict['corr_mat'] = []
for run_idx, run in enumerate(regressors_dict['regressors']):
    sub = regressors_dict['subject'][run_idx]
    ses = regressors_dict['session'][run_idx]
    run = regressors_dict['run'][run_idx]
    confounds_fname = op.join(path_to_data, 'shinobi',
                                            'derivatives',
                                            'fmriprep-20.2lts',
                                            'fmriprep',
                                            sub, ses,
                                            'func',
                                            f'{sub}_{ses}_task-shinobi_run-{run}_desc-confounds_timeseries.tsv')
    corr = regressors_dict['regressors'][run_idx].corr()
    regressors_dict['corr_mat'].append(corr)

for sub in subjects:
    subj_corrs = []
    for idx, corr_mat in enumerate(regressors_dict['corr_mat']):
        if regressors_dict['subject'][idx] == sub:
            subj_corrs.append(corr_mat)

    averaged_corr_mat = pd.concat(subj_corrs, axis=0).groupby(level=0).mean()


    mask = np.triu(np.ones_like(averaged_corr_mat, dtype=bool))
    f, ax = plt.subplots(figsize=(30, 25))

    sns.heatmap(averaged_corr_mat, mask=mask, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5}).set_title(f'{sub}')
    fig_fname = op.join(figures_path, 'design_matrices', f'regressor_correlations_{sub}.png')
    plt.savefig(fig_fname)
    plt.close()


    # Scatter + density plots
