"""
Generate correlation matrices for GLM design matrix regressors.

This script computes and visualizes correlation matrices between all regressors
(annotations) in the GLM design matrices. It produces:
1. Per-run correlation heatmaps and clustermaps
2. Subject-averaged correlation heatmaps and clustermaps

The correlations help identify multicollinearity issues and understand
relationships between different task events.
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
from load_confounds import Confounds

import shinobi_fmri.config as config
from shinobi_fmri.annotations.annotations import trim_events_df
from shinobi_fmri.utils.logger import ShinobiLogger
from shinobi_fmri.glm.compute_run_level import add_psychophysics_confounds, add_button_press_confounds

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
        help="Include low-level confounds (psychophysics and button presses) in design matrix",
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

        # Load and trim events first
        run_events = pd.read_csv(events_fname, sep='\t', low_memory=False)
        events_df = trim_events_df(run_events, trim_by="event")

        # Load confounds
        confounds_obj = Confounds(
            strategy=["motion", 'global', 'wm_csf'],
            motion="full",
            wm_csf='basic',
            global_signal='full'
        )
        try:
            confounds_df = confounds_obj.load(fmri_fname)
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
            confounds_tuple = (confounds_df,)
            confounds_tuple = add_psychophysics_confounds(confounds_tuple, run_events)
            confounds_tuple = add_button_press_confounds(confounds_tuple, run_events)
            confounds = confounds_tuple[0]
        else:
            confounds = confounds_df

        # Generate design matrix
        n_slices = confounds.shape[0]
        frame_times = np.arange(n_slices) * t_r

        # Filter events to keep only relevant columns and avoid warnings
        events_cols = ['onset', 'duration', 'trial_type']
        if 'modulation' in events_df.columns:
            events_cols.append('modulation')
        events_df_clean = events_df[events_cols]

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
            'run': run,
            'confounds_shape': confounds.shape
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
        logger: ShinobiLogger instance

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
                logger.debug(f"Confounds shape: {res['confounds_shape']}")
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
        logger: ShinobiLogger instance
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

        # Cluster map
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
        logger: ShinobiLogger instance
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

        # Cluster map
        f = sns.clustermap(
            averaged_corr_mat, mask=mask, figsize=(30, 25),
            center=0, square=True, linewidths=.5, annot=True,
            cbar_kws={"shrink": .5}, fmt='.2f'
        )
        fig_fname = op.join(output_dir, f'regressor_correlations_{sub}_cluster.png')
        f.savefig(fig_fname, dpi=100, bbox_inches='tight')
        plt.close()

        if logger:
            logger.summary.add_computed(f"Subject-averaged correlations: {sub}")


def main():
    """Main execution function."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup logging
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    verbosity_level = verbosity_map.get(args.verbose, logging.DEBUG)
    logger = ShinobiLogger(
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

    # Generate correlation plots
    logger.info("Plotting run-level correlations...")
    plot_run_correlations(regressors_dict, figures_path, logger)

    logger.info("Plotting subject-averaged correlations...")
    plot_subject_averaged_correlations(regressors_dict, subjects, figures_path, logger)

    logger.info("Done!")
    logger.close()


if __name__ == "__main__":
    main()
