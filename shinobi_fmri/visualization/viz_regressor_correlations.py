"""
Generate design matrices and correlation visualization for GLM regressors.

This script computes and visualizes design matrices and correlations between
key regressors in the GLM. It produces:
1. Design matrices (one per run) showing all regressors
2. 2x2 correlation grid showing subject-averaged correlations

The 2x2 grid focuses on Shinobi task conditions and low-level confounds
(psychophysics: luminance, optical_flow, audio_envelope; and button press:
button_presses_count) to identify multicollinearity between task events and
low-level visual/audio features and motor activity.
"""

import os
import os.path as op
import argparse
import pickle
import logging
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.signal import clean
import nilearn.interfaces.fmriprep

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.glm.utils import add_psychophysics_confounds, add_button_press_confounds

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def setup_argparse():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate correlation matrices for design matrix regressors"
    )
    parser.add_argument(
        "-s", "--subject",
        default=None,
        type=str,
        help="Specific subject to process (default: all subjects)",
    )
    parser.add_argument(
        "--data-path",
        default=config.path_to_data,
        type=str,
        help=f"Path to data directory (default: {config.path_to_data})",
    )
    parser.add_argument(
        "--figures-path",
        default=config.figures_path,
        type=str,
        help=f"Path to figures output directory (default: {config.figures_path})",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip design matrix generation, only plot from existing pickle file",
    )
    parser.add_argument(
        "--low-level-confs",
        action="store_true",
        default=True,
        help="Include low-level confounds (psychophysics and button presses) in design matrix (default: True)",
    )
    parser.add_argument(
        "--no-low-level-confs",
        dest="low_level_confs",
        action="store_false",
        help="Exclude low-level confounds from design matrix",
    )
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
    return parser


def process_single_run(sub, ses, run, path_to_data, figures_path, use_low_level_confs=False):
    """Helper function to process a single run."""
    t_r = 1.49
    hrf_model = 'spm'
    
    try:
        # Zero-pad run number for events filename (run-01 format)
        run_padded = run.zfill(2)

        # Build file paths
        events_fname = op.join(
            path_to_data, "shinobi", sub, ses, "func",
            f"{sub}_{ses}_task-shinobi_run-{run_padded}_desc-annotated_events.tsv"
        )
        fmri_fname = op.join(
            path_to_data, "shinobi.fmriprep",
            sub, ses, "func",
            f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        )

        # Check if events file exists
        if not op.exists(events_fname):
            return {
                'success': False,
                'subject': sub,
                'session': ses,
                'run': run,
                'error': f"Events file not found: {events_fname}"
            }

        # Load events
        run_events = pd.read_csv(events_fname, sep='\t', low_memory=False)

        # Load confounds using nilearn interface (returns tuple)
        try:
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                fmri_fname,
                strategy=("motion", "high_pass", "wm_csf"),
                motion="full",
                wm_csf="basic",
                global_signal="full",
            )
        except Exception as e:
             return {
                'success': False,
                'subject': sub,
                'session': ses,
                'run': run,
                'error': f"Could not load confounds (broken link or missing file): {str(e)}"
            }

        # Add low-level confounds if requested
        if use_low_level_confs:
            confounds = add_psychophysics_confounds(confounds, run_events, path_to_data, t_r=t_r)
            confounds = add_button_press_confounds(confounds, run_events, t_r=t_r)

        # Extract DataFrame from tuple
        confounds = confounds[0]

        # Generate design matrix
        n_slices = confounds.shape[0]
        frame_times = np.arange(n_slices) * t_r

        # Filter events to keep only relevant columns and avoid warnings
        events_cols = ['onset', 'duration', 'trial_type']
        if 'modulation' in run_events.columns:
            events_cols.append('modulation')
        events_df_clean = run_events[events_cols]

        design_matrix_raw = make_first_level_design_matrix(
            frame_times,
            events=events_df_clean,
            drift_model=None,
            hrf_model=hrf_model,
            add_regs=confounds,
            add_reg_names=None
        )

        # Clean design matrix
        regressors_clean = clean(
            design_matrix_raw.to_numpy(),
            detrend=True,
            standardize='zscore_sample',
            high_pass=None,
            t_r=t_r,
            ensure_finite=True,
            confounds=None,
        )
        design_matrix_clean = pd.DataFrame(
            regressors_clean,
            columns=design_matrix_raw.columns.to_list()
        )

        # Save design matrix plot
        from nilearn import plotting as nlplot
        design_matrix_clean_fname = op.join(
            figures_path, "design_matrices",
            f"design_matrix_clean_{sub}_{ses}_run-{run_padded}.png",
        )
        os.makedirs(op.dirname(design_matrix_clean_fname), exist_ok=True)
        nlplot.plot_design_matrix(
            design_matrix_clean,
            output_file=design_matrix_clean_fname
        )
        
        return {
            'success': True,
            'regressors': design_matrix_clean,
            'subject': sub,
            'session': ses,
            'run': run
        }

    except Exception as e:
        return {
            'success': False,
            'subject': sub,
            'session': ses,
            'run': run,
            'error': str(e)
        }


def build_design_matrices(subjects, path_to_data, figures_path, use_low_level_confs=False, logger=None):
    """
    Build design matrices for all runs across specified subjects.

    Args:
        subjects: List of subject IDs to process
        path_to_data: Path to data directory
        figures_path: Path to figures directory
        use_low_level_confs: Include low-level confounds (psychophysics and button presses)
        logger: AnalysisLogger instance

    Returns:
        dict: Dictionary containing design matrices, subjects, sessions, and runs
    """
    regressors_dict = {
        'regressors': [],
        'subject': [],
        'session': [],
        'run': []
    }

    # Collect all tasks to run
    tasks = []
    
    # Pre-scan for tasks
    for sub in subjects:
        fmriprep_sub_path = op.join(path_to_data, "shinobi.fmriprep", sub)
        if not op.exists(fmriprep_sub_path):
            if logger:
                logger.warning(f"Subject directory not found: {fmriprep_sub_path}")
            continue

        sessions = [d for d in os.listdir(fmriprep_sub_path)
                   if d.startswith('ses-') and op.isdir(op.join(fmriprep_sub_path, d))]

        for ses in sorted(sessions):
            ses_fpath = op.join(path_to_data, "shinobi.fmriprep", sub, ses, "func")
            if not op.exists(ses_fpath):
                if logger:
                    logger.warning(f"Func directory not found: {ses_fpath}")
                continue

            ses_files = os.listdir(ses_fpath)
            run_files = [
                x for x in ses_files
                if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in x
            ]

            # Extract run numbers
            run_list = []
            for fname in run_files:
                match = re.search(r'run-(\d+)', fname)
                if match:
                    run_list.append(match.group(1))

            for run in sorted(run_list):
                tasks.append((sub, ses, run))

    # Process sequentially
    if logger:
        logger.info(f"Processing {len(tasks)} runs...")

    for sub, ses, run in tqdm(tasks, desc="Building design matrices"):
        res = process_single_run(
            sub, ses, run, path_to_data, figures_path, use_low_level_confs
        )
        if res['success']:
            regressors_dict['regressors'].append(res['regressors'])
            regressors_dict['subject'].append(res['subject'])
            regressors_dict['session'].append(res['session'])
            regressors_dict['run'].append(res['run'])

            if logger:
                logger.summary.add_computed(f"Design matrix: {res['subject']} {res['session']} run {res['run']}")
        else:
            if logger:
                logger.error(f"Error processing {res['subject']} {res['session']} run {res['run']}: {res['error']}")
                logger.summary.add_error(f"{res['subject']} {res['session']} run {res['run']}", res['error'])
            else:
                print(f"Error processing {res['subject']} {res['session']} run {res['run']}: {res['error']}")

    return regressors_dict


def plot_run_correlations(regressors_dict, figures_path, logger=None):
    """
    Plot correlation matrices for each run.

    Args:
        regressors_dict: Dictionary containing design matrices
        figures_path: Path to save figures
        logger: AnalysisLogger instance
    """
    output_dir = op.join(figures_path, 'design_matrices')
    os.makedirs(output_dir, exist_ok=True)

    regressors_dict['corr_mat'] = []

    pbar = tqdm(enumerate(regressors_dict['regressors']),
                total=len(regressors_dict['regressors']),
                desc="Plotting run correlations")

    for run_idx, run in pbar:
        sub = regressors_dict['subject'][run_idx]
        ses = regressors_dict['session'][run_idx]
        run_num = regressors_dict['run'][run_idx]

        if logger:
            logger.info(f"Plotting correlations for {sub} {ses} run {run_num}")

        # Compute correlation matrix (excluding constant)
        corr = regressors_dict['regressors'][run_idx].corr()
        if 'constant' in corr.index:
            corr = corr.drop(index='constant').drop(columns='constant')
        regressors_dict['corr_mat'].append(corr)

        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Heatmap
        f, ax = plt.subplots(figsize=(30, 25))
        sns.heatmap(
            corr, mask=mask, center=0,
            square=True, linewidths=.5, annot=False,
            cbar_kws={"shrink": .5}
        ).set_title(f'{sub} {ses} run {run_num}')
        fig_fname = op.join(
            output_dir,
            f'regressor_correlations_{sub}_{ses}_run-{run_num}.png'
        )
        plt.savefig(fig_fname, dpi=100, bbox_inches='tight')
        plt.close()

        # Cluster map (only if no NaN/inf values)
        if np.all(np.isfinite(corr.values)):
            try:
                f = sns.clustermap(
                    corr, mask=mask, figsize=(30, 25),
                    cbar_kws={"shrink": .5}
                )
                fig_fname = op.join(
                    output_dir,
                    f'regressor_correlations_{sub}_{ses}_run-{run_num}_cluster.png'
                )
                f.savefig(fig_fname, dpi=100, bbox_inches='tight')
                plt.close()
            except ValueError as e:
                if logger:
                    logger.warning(f"Could not create clustermap for {sub} {ses} run {run_num}: {str(e)}")

        if logger:
            logger.summary.add_computed(f"Run correlations: {sub} {ses} run {run_num}")

        pbar.set_postfix({"Subject": sub, "Session": ses, "Run": run_num})


def plot_subject_averaged_correlations(regressors_dict, subjects, figures_path, logger=None):
    """
    Plot subject-averaged correlation matrices.

    Args:
        regressors_dict: Dictionary containing correlation matrices
        subjects: List of subjects to process
        figures_path: Path to save figures
        logger: AnalysisLogger instance
    """
    output_dir = op.join(figures_path, 'design_matrices')
    os.makedirs(output_dir, exist_ok=True)

    for sub in tqdm(subjects, desc="Plotting subject-averaged correlations"):
        if logger:
            logger.info(f"Plotting averaged correlations for {sub}")

        # Collect all correlation matrices for this subject
        subj_corrs = []
        for idx, corr_mat in enumerate(regressors_dict['corr_mat']):
            if regressors_dict['subject'][idx] == sub:
                subj_corrs.append(corr_mat)

        if not subj_corrs:
            if logger:
                logger.warning(f"No correlation matrices found for {sub}")
            continue

        # Average across runs
        averaged_corr_mat = pd.concat(subj_corrs, axis=0).groupby(level=0).mean()
        mask = np.triu(np.ones_like(averaged_corr_mat, dtype=bool))

        # Heatmap with annotations
        f, ax = plt.subplots(figsize=(30, 25))
        sns.heatmap(
            averaged_corr_mat, mask=mask, center=0,
            square=True, linewidths=.5, annot=True,
            cbar_kws={"shrink": .5}, fmt='.2f'
        ).set_title(f'{sub} - Averaged across runs')
        fig_fname = op.join(output_dir, f'regressor_correlations_{sub}.png')
        plt.savefig(fig_fname, dpi=100, bbox_inches='tight')
        plt.close()

        # Cluster map (only if no NaN/inf values)
        if np.all(np.isfinite(averaged_corr_mat.values)):
            try:
                f = sns.clustermap(
                    averaged_corr_mat, mask=mask, figsize=(30, 25),
                    center=0, square=True, linewidths=.5, annot=True,
                    cbar_kws={"shrink": .5}, fmt='.2f'
                )
                fig_fname = op.join(output_dir, f'regressor_correlations_{sub}_cluster.png')
                f.savefig(fig_fname, dpi=100, bbox_inches='tight')
                plt.close()
            except ValueError as e:
                if logger:
                    logger.warning(f"Could not create clustermap for {sub}: {str(e)}")
        else:
            if logger:
                logger.warning(f"Skipping clustermap for {sub} due to non-finite correlation values")

        if logger:
            logger.summary.add_computed(f"Subject-averaged correlations: {sub}")


def plot_2x2_subject_correlations(regressors_dict, subjects, figures_path, logger=None):
    """
    Plot a 2x2 grid of subject-averaged correlation matrices.

    Creates a compact visualization showing the lower triangle of each subject's
    averaged correlation matrix in a 2x2 layout with a shared red-white-blue colorbar.

    Only includes 8 Shinobi task conditions (DOWN, HIT, HealthGain, HealthLoss,
    JUMP, Kill, LEFT, RIGHT), 3 psychophysics confounds (luminance,
    optical_flow, audio_envelope), and button press confounds (button_presses_count).
    Correlation values are displayed in each cell with 2 decimal places.

    Args:
        regressors_dict: Dictionary containing correlation matrices
        subjects: List of subjects to process (should be 4 subjects for 2x2 grid)
        figures_path: Path to save figures
        logger: AnalysisLogger instance
    """
    output_dir = op.join(figures_path, 'design_matrices')
    os.makedirs(output_dir, exist_ok=True)

    if logger:
        logger.info(f"Plotting 2x2 grid for {len(subjects)} subjects")

    # Define which regressors to include
    # Shinobi conditions (excluding UP - not analyzed)
    shinobi_conditions = ['DOWN', 'HIT', 'HealthGain', 'HealthLoss', 'JUMP',
                          'Kill', 'LEFT', 'RIGHT']
    # Psychophysics confounds
    psychophysics_confounds = ['luminance', 'optical_flow', 'audio_envelope']
    # Button press confounds
    button_confounds = ['button_presses_count']
    # Combined list
    regressors_to_include = shinobi_conditions + psychophysics_confounds + button_confounds

    # Prepare averaged correlation matrices for each subject
    subject_corr_mats = {}
    all_values = []  # To compute global vmin/vmax

    for sub in subjects:
        # Collect all correlation matrices for this subject
        subj_corrs = []
        for idx, corr_mat in enumerate(regressors_dict['corr_mat']):
            if regressors_dict['subject'][idx] == sub:
                subj_corrs.append(corr_mat)

        if not subj_corrs:
            if logger:
                logger.warning(f"No correlation matrices found for {sub}")
            continue

        # Average across runs
        averaged_corr_mat = pd.concat(subj_corrs, axis=0).groupby(level=0).mean()

        # Filter to keep only specified regressors
        available_regressors = [r for r in regressors_to_include if r in averaged_corr_mat.index]
        if len(available_regressors) == 0:
            if logger:
                logger.warning(f"No target regressors found for {sub}")
            continue

        averaged_corr_mat = averaged_corr_mat.loc[available_regressors, available_regressors]
        subject_corr_mats[sub] = averaged_corr_mat

        # Collect values from lower triangle for global color scale
        mask_lower = np.tril(np.ones_like(averaged_corr_mat, dtype=bool), k=-1)
        all_values.extend(averaged_corr_mat.values[mask_lower])

    if not subject_corr_mats:
        if logger:
            logger.error("No correlation matrices to plot")
        return

    # Compute global color scale
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    abs_max = max(abs(vmin), abs(vmax))

    # Create 2x2 figure with larger size for bigger cells
    fig, axes = plt.subplots(2, 2, figsize=(28, 26))
    axes = axes.flatten()

    # Plot each subject in a subplot
    for idx, sub in enumerate(subjects[:4]):  # Only plot first 4 subjects
        ax = axes[idx]

        if sub not in subject_corr_mats:
            ax.set_visible(False)
            continue

        corr_mat = subject_corr_mats[sub]

        # Create mask for upper triangle (including diagonal)
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))

        # Plot heatmap with RdBu_r colormap (red-white-blue)
        sns.heatmap(
            corr_mat,
            mask=mask,
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            cmap='RdBu_r',
            square=True,
            linewidths=0.5,
            cbar=False,  # We'll add a shared colorbar later
            ax=ax,
            xticklabels=True,
            yticklabels=True,
            annot=True,  # Show correlation values in squares
            fmt='.2f',  # Two decimal places
            annot_kws={'fontsize': 16}  # Font size for annotations (doubled from 8)
        )

        # Subtitle (subject name) - not bold, positioned very close to matrix
        # Use ax.text() to manually position the subtitle near the top of the axes
        ax.text(0.5, 0.92, f'{sub}', transform=ax.transAxes,
                fontsize=32, fontweight='normal', ha='center', va='top')

        # Get tick labels
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()

        # Remove first row label (no data shown) and last column label (no data shown)
        # since we mask the diagonal
        yticklabels[0].set_visible(False)  # First row (DOWN)
        xticklabels[-1].set_visible(False)  # Last column (audio_envelope)

        # Rotate labels for better readability with slightly smaller font
        ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=18)
        ax.set_yticklabels(yticklabels, rotation=0, fontsize=18)

    # Hide unused subplots if less than 4 subjects
    for idx in range(len(subjects), 4):
        axes[idx].set_visible(False)

    # Add shared colorbar on the right (size of one subplot, vertically centered)
    fig.subplots_adjust(right=0.92)
    # Calculate vertical center and height to match one subplot
    subplot_height = 0.35  # Approximate height of one subplot in figure coordinates
    vertical_center = 0.5  # Center of figure
    cbar_bottom = vertical_center - (subplot_height / 2)
    cbar_ax = fig.add_axes([0.94, cbar_bottom, 0.02, subplot_height])
    sm = plt.cm.ScalarMappable(
        cmap='RdBu_r',
        norm=plt.Normalize(vmin=-abs_max, vmax=abs_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Set colorbar tick label size (doubled from default)
    cbar.ax.tick_params(labelsize=20)

    # Add "r̄" (r bar) label above the colorbar (not rotated)
    cbar.ax.text(0.5, 1.05, r'$\bar{r}$', transform=cbar.ax.transAxes,
                 fontsize=36, ha='center', va='bottom')

    # Save figure
    fig_fname = op.join(output_dir, 'regressor_correlations_2x2_subjects.png')
    plt.savefig(fig_fname, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Saved 2x2 correlation plot to {fig_fname}")
        logger.summary.add_computed("2x2 subject correlation grid")


def main():
    """Main execution function."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup logging
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    verbosity_level = verbosity_map.get(args.verbose, logging.DEBUG)
    logger = AnalysisLogger(
        log_name="regressor_correlations",
        verbosity=verbosity_level,
        log_dir=args.log_dir
    )

    # Configuration
    figures_path = args.figures_path
    path_to_data = args.data_path
    subjects = [args.subject] if args.subject else config.subjects

    logger.info(f"Processing subjects: {subjects}")
    logger.info(f"Data path: {path_to_data}")
    logger.info(f"Figures path: {figures_path}")

    # Path for pickle file
    regressors_dict_fname = op.join(
        path_to_data, "processed", "regressors_dict.pkl"
    )

    # Build or load design matrices
    if args.skip_generation and op.exists(regressors_dict_fname):
        logger.info(f"Loading existing regressors from {regressors_dict_fname}")
        with open(regressors_dict_fname, "rb") as f:
            regressors_dict = pickle.load(f)
        logger.summary.add_skipped("Design matrix generation (loaded from pickle)")
    else:
        logger.info("Building design matrices...")
        if args.low_level_confs:
            logger.info("Including low-level confounds (psychophysics and button presses)")
        regressors_dict = build_design_matrices(
            subjects, path_to_data, figures_path,
            use_low_level_confs=args.low_level_confs,
            logger=logger
        )

        # Save to pickle
        os.makedirs(op.dirname(regressors_dict_fname), exist_ok=True)
        logger.info(f"Saving regressors to {regressors_dict_fname}")
        with open(regressors_dict_fname, "wb") as f:
            pickle.dump(regressors_dict, f)

    # Compute correlation matrices (needed for 2x2 plot)
    logger.info("Computing correlation matrices...")
    regressors_dict['corr_mat'] = []
    for run_idx, run in enumerate(regressors_dict['regressors']):
        # Compute correlation matrix (excluding constant)
        corr = run.corr()
        if 'constant' in corr.index:
            corr = corr.drop(index='constant').drop(columns='constant')
        regressors_dict['corr_mat'].append(corr)

    # Generate 2x2 subject correlation grid (Shinobi conditions + 3 psychophysics confounds only)
    logger.info("Plotting 2x2 subject correlation grid...")
    plot_2x2_subject_correlations(regressors_dict, subjects, figures_path, logger)

    logger.info("Done!")
    logger.close()


if __name__ == "__main__":
    main()
