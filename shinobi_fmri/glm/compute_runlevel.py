import os
import os.path as op
import pandas as pd
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import get_scrub_regressor
import numpy as np
import pdb
import argparse
import nilearn
import shinobi_behav
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.image import clean_img
from nilearn.reporting import get_clusters_table
from nilearn import input_data
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.signal import clean
import nibabel as nib
import logging
import pickle
from nilearn.plotting import plot_img_on_surf, plot_stat_map
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-01",
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-ses",
    "--session",
    default="ses-013",
    type=str,
    help="Session to process",
)
parser.add_argument(
    "-r",
    "--run",
    default="1",
    type=str,
    help="Run to process",
)
args = parser.parse_args()




def get_filenames(sub, ses, run, path_to_data):
    fmri_fname = op.join(
        path_to_data,
        "shinobi.fmriprep",
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    assert op.isfile(fmri_fname), f"fMRI file not found for {sub}_{ses}_{run}"

    anat_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    assert op.isfile(anat_fname), f"sMRI file not found for {sub}_{ses}_{run}"

    events_fname = op.join(
        path_to_data, 
        "shinobi", 
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-0{run}_annotated_events.tsv"
    )
    assert op.isfile(events_fname), f"Annotated events file not found for {sub}_{ses}_{run}" 

    glm_fname = op.join(path_to_data,
                    "processed25122022",
                    "glm",
                    "run-level",
                    f"{sub}_{ses}_run-0{run}_fitted_glm.pkl")

    os.makedirs(op.join(path_to_data,
                        "processed25122022",
                        "glm",
                        "run-level"), exist_ok=True)

    return fmri_fname, anat_fname, events_fname, glm_fname


def process_run(fmri_fname, anat_fname, events_fname, glm_fname, savefigs=True):
    # Load events
    run_events = pd.read_csv(events_fname, sep="\t", index_col=[0])

    # Select events
    annotation_events = run_events[run_events["trial_type"].isin(['B', 'C', 'DOWN', 'HealthGain', 'HealthLoss', 'Kill', 'LEFT',
        'RIGHT', 'UP'])] # add 'frame' here
    annotation_events = annotation_events[["trial_type", "onset", "duration"]]

    # Load and clean 4d image
    fmri_img = clean_img(
        fmri_fname,
        standardize=True,
        detrend=True,
        high_pass=None,
        t_r=t_r,
        ensure_finite=True,
        confounds=None,
    )

    # Load and resample (i.e. morph  ?) anat mask
    aff_orig = nib.load(fmri_fname).affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    anat_img = image.resample_img(anat_fname, target_affine=target_affine, target_shape=fmri_img.get_fdata().shape[:3])

    # Make design matrix
    # Load confounds
    confounds = Confounds(
        strategy=["high_pass", "motion", "global", "wm_csf"],
        motion="full",
        wm_csf="basic",
        global_signal="full",
    ).load(fmri_fname)

    # Generate design matrix
    bold_shape = fmri_img.shape
    n_slices = bold_shape[-1]
    frame_times = np.arange(n_slices) * t_r
    design_matrix_raw = make_first_level_design_matrix(
        frame_times,
        events=annotation_events,
        drift_model=None,
        hrf_model=hrf_model,
        add_regs=confounds,
        add_reg_names=None,
    )

    # Clean regressors 
    regressors_clean = clean(
        design_matrix_raw.to_numpy(),
        detrend=True,
        standardize=True,
        high_pass=None,
        t_r=t_r,
        ensure_finite=True,
        confounds=None,
    )

    # Recombine design_matrix (restoring constant after cleaning and adding scrub regressors)
    design_matrix_clean = pd.DataFrame(
        regressors_clean, columns=design_matrix_raw.columns.to_list()
    )
    design_matrix_clean["constant"] = 1
    design_matrix_clean = get_scrub_regressor(run_events, design_matrix_clean)

    if savefigs:
        design_matrix_raw_fname = op.join(
            figures_path,
            "design_matrices",
            f"design_matrix_raw_{sub}_{ses}_run-0{run}.png",
        )
        plotting.plot_design_matrix(
            design_matrix_raw, output_file=design_matrix_raw_fname
        )
        design_matrix_clean_fname = op.join(
            figures_path,
            "design_matrices",
            f"design_matrix_clean_{sub}_{ses}_run-0{run}.png",
        )
        plotting.plot_design_matrix(
            design_matrix_clean, output_file=design_matrix_clean_fname
        )

    # Fit GLM
    fmri_glm = FirstLevelModel(
        t_r=t_r,
        noise_model="ar1",
        standardize=False,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=None,
        n_jobs=16,
        smoothing_fwhm=5,
        mask_img=anat_img,
        minimize_memory=False,
    )
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix_clean)

    with open(glm_fname, "wb") as f:
        pickle.dump(fmri_glm, f)

    return fmri_glm




def main():

    fmri_fname, anat_fname, events_fname, glm_fname = get_filenames(sub, ses, run, path_to_data)
    fmri_glm = process_run(fmri_fname, anat_fname, events_fname, glm_fname, savefigs=True)


    

if __name__ == "__main__":
    figures_path = shinobi_behav.FIG_PATH #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.DATA_PATH  #'/media/storage/neuromod/shinobi_data/'
    sub = args.subject
    ses = args.session
    run = args.run
    t_r = 1.49
    hrf_model = "spm"
    # Log job info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    print(f"Processing : {sub} {ses} {run}")
    print(figures_path)
    print(path_to_data)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()