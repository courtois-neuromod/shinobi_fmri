import os
import os.path as op
import pandas as pd
from nilearn import image, signal
from load_confounds import Confounds
from shinobi_fmri.annotations.annotations import trim_events_df, get_scrub_regressor
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="01",
    type=str,
    help="Subject to process",
)
args = parser.parse_args()


def create_output_folders(path_to_data, figures_path, contrasts):
    # Create output folders if needed
    os.makedirs(op.join(figures_path, "design_matrices"), exist_ok=True)
    os.makedirs(op.join(path_to_data, "processed", "run-level"), exist_ok=True)
    for contrast in contrasts:
        os.makedirs(
            op.join(path_to_data, "processed", "z_maps", "run-level", contrast), exist_ok=True)
        os.makedirs(op.join(figures_path, "run-level", contrast), exist_ok=True)


def compute_runlevel_glm(sub, ses, run, t_r=1.49, hrf_model="spm", savefigs=True):
    # Generate fnames
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
    anat_fname = op.join(
        path_to_data,
        "anat",
        "derivatives",
        "fmriprep-20.2lts",
        "fmriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    events_fname = op.join(
        path_to_data, "processed", "annotations", f"{sub}_{ses}_run-0{run}.csv"
    )

    run_events = pd.read_csv(events_fname)
    if run_events.empty:
        print("run_events is empty")
    else:
        events_df = trim_events_df(run_events, trim_by="event")
        # Load inputs
        confounds = Confounds(
            strategy=["high_pass", "motion"],
            motion="full",
            wm_csf="full",
            global_signal="full",
        ).load(fmri_fname)
        fmri_img = clean_img(
            fmri_fname,
            standardize=True,
            detrend=True,
            high_pass=None,
            t_r=t_r,
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
            add_regs=confounds,
            add_reg_names=None,
        )
        regressors_clean = clean(
            design_matrix_raw.to_numpy(),
            detrend=True,
            standardize=True,
            high_pass=None,
            t_r=t_r,
            ensure_finite=True,
            confounds=None,
        )
        design_matrix_clean = pd.DataFrame(
            regressors_clean, columns=design_matrix_raw.columns.to_list()
        )
        design_matrix_clean["constant"] = 1
        # design_matrix_clean = get_scrub_regressor(run_events, design_matrix_clean) # TODO : one regressor per scrubbed volume

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
            mask_img=anat_fname,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix_clean)
    return fmri_glm, fmri_img, fmri_fname


def plot_signals(z_map, fmri_img, fmri_fname):
    # Load background image
    mean_img = image.mean_img(fmri_fname)
    # Extract activation clusters
    table = get_clusters_table(
        z_map, stat_threshold=3.1, cluster_threshold=20
    ).set_index("Cluster ID", drop=True)
    # get the 3 largest clusters' max x, y, and z coordinates
    coords = table.loc[range(1, 4), ["X", "Y", "Z"]].values
    # extract time series from each coordinate
    masker = input_data.NiftiSpheresMasker(coords)
    real_timeseries = masker.fit_transform(fmri_img)
    predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])
    test_statistic = masker.fit_transform(z_map)

    # Plot figure
    # colors for each of the clusters
    colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
    # plot the time series and corresponding locations
    fig_signals, axs = plt.subplots(2, 3)
    for i in range(0, 3):
        # plotting time series
        axs[0, i].set_title(
            "Cluster peak {}\n".format(coords[i])
        )  # TODO : replace by test_statistic
        axs[0, i].plot(real_timeseries[:, i], c=colors[i], lw=2)
        axs[0, i].plot(predicted_timeseries[:, i], c="r", ls="--", lw=2)
        axs[0, i].set_xlabel("Time")
        axs[0, i].set_ylabel("Signal intensity", labelpad=0)
        # plotting image below the time series
        roi_img = plotting.plot_stat_map(
            z_map,
            cut_coords=[coords[i][2]],
            threshold=3.1,
            figure=fig1,
            axes=axs[1, i],
            display_mode="z",
            colorbar=False,
            bg_img=mean_img,
        )
        roi_img.add_markers([coords[i]], colors[i], 300)

    fig_signals.set_size_inches(24, 14)
    return fig_signals


def process_run(sub, ses, run):
    print(f"Processing run 0{run}")
    glm_fname = op.join(
        path_to_data, "processed", "run-level", f"glm_{sub}_{ses}_run-0{run}.pkl"
    )
    if os.path.exists(glm_fname):
        print("GLM already exists")
        with open(glm_fname, "rb") as f:
            fmri_glm = pickle.load(f)
            print("GLM loaded")
    else:
        try:
            print("Computing GLM")
            fmri_glm, fmri_img, fmri_fname = compute_runlevel_glm(
                sub, ses, run, t_r=t_r, hrf_model=hrf_model, savefigs=True
            )
            with open(glm_fname, "wb") as f:
                pickle.dump(fmri_glm, f)
            print("GLM saved")
        except Exception as e:
            print("GLM not computed")
            print(e)
            return
        for contrast in contrasts:
            print(f"Computing contrast : {contrast}")
            # Compute contrast
            z_map_fname = op.join(
                path_to_data,
                "processed",
                "z_maps",
                "run-level",
                contrast,
                f"{sub}_{ses}_run-0{run}.nii.gz",
            )
            if not (os.path.exists(z_map_fname)):
                z_map = fmri_glm.compute_contrast(
                    contrast, output_type="z_score", stat_type="F"
                )
                z_map.to_filename(z_map_fname)
                # Save report
                report_fname = op.join(
                    figures_path,
                    "run-level",
                    contrast,
                    f"{sub}_{ses}_run-0{run}_{contrast}_flm.html",
                )
                report = fmri_glm.generate_report(contrasts=[contrast])
                report.save_as_html(report_fname)
                # Save images
                print("Generating views")
                # FDR corrected image
                clean_map, threshold = threshold_stats_img(
                    z_map, alpha=0.05, height_control="fdr", cluster_threshold=10
                )
                view = plotting.view_img(
                    clean_map,
                    threshold=3,
                    title=f"{contrast} (FDR<0.05), Noyaux > 10 voxels",
                )
                view.save_as_html(
                    op.join(
                        figures_path,
                        "run-level",
                        contrast,
                        f"{sub}_{ses}_run-0{run}_{contrast}_flm_FDRcluster_fwhm5.html",
                    )
                )
                # Uncorrected image
                uncorr_map, threshold = threshold_stats_img(
                    z_map, alpha=0.001, height_control="fpr"
                )
                view = plotting.view_img(
                    uncorr_map, threshold=3, title=f"{contrast} (p<0.001), uncorr"
                )
                view.save_as_html(
                    op.join(
                        figures_path,
                        "run-level",
                        contrast,
                        f"{sub}_{ses}_run-0{run}_{contrast}_flm_uncorr_fwhm5.html",
                    )
                )
                # Observed VS predicted values for top clusters
                fig_signals = plot_signals(z_map, fmri_img, fmri_fname)
                signals_plot_name = op.join(
                    figures_path,
                    "run-level",
                    contrast,
                    f"signals_{sub}_{ses}_run-0{run}_{contrast}.png",
                )
                fig_signals.savefig(signals_plot_name)
                print("Done")


def main():
    create_output_folders(path_to_data, figures_path, contrasts)
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
            process_run(sub, ses, run)


if __name__ == "__main__":
    # Set constants
    figures_path = (
        shinobi_behav.figures_path
    )  #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
    path_to_data = shinobi_behav.path_to_data  #'/media/storage/neuromod/shinobi_data/'
    sub = "sub-" + args.subject
    t_r = 1.49
    hrf_model = "spm"
    contrasts = ["Hit", "Kill", "Jump"]

    # Log job info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
