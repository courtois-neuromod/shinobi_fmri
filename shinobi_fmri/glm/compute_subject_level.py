"""
Subject-Level (Third-Level) GLM Analysis for fMRI Data

This script performs subject-level (third-level) GLM analysis by combining
session-level z-maps across multiple sessions for a single subject.

STATISTICAL METHODS:
-------------------
GLM Specification:
  - Model: Second-level GLM (random-effects across sessions)
  - Analysis level: Within-subject, combining session-level maps
  - No additional smoothing (session-level maps already smoothed)
  - Noise model: Ordinary Least Squares (between-session variance)

Design Matrix:
  - Single regressor: Intercept (one-sample t-test across sessions)
  - No covariates (pure averaging of session effects)
  - Input: Session-level z-maps for specified contrast

Statistical Inference:
  - Test type: One-sample F-test (mean activation across sessions)
  - Null hypothesis: No consistent activation across sessions
  - Multiple comparison correction: Cluster-level FWE correction
  - Cluster-forming threshold: Z > 3.1 (conservative for subject-level)
  - Family-wise error rate: alpha = 0.05
  - Effect: Random-effects estimate of consistent activation

Contrast:
  - Type: Intercept contrast (mean across sessions)
  - Interpretation: Brain regions consistently active for this condition
  - Output: Z-scores testing mean ≠ 0 across sessions

Outputs:
  - Beta maps: Subject-level effect sizes (mean across sessions)
  - Z-maps: Uncorrected F-test z-scores
  - Corrected Z-maps: Cluster-corrected statistical maps
  - HTML reports: Interactive visualizations with glass brains
  - Metadata JSON: Complete provenance tracking with session list

References:
  - Holmes & Friston (1998). Generalisability, random effects and population inference.
    NeuroImage, 7(4), S754.
  - Beckmann et al. (2003). General multilevel linear modeling for group analysis in FMRI.
    NeuroImage, 20(2), 1052-1063.

USAGE:
------
  python compute_subject_level.py -s sub-01 -cond HIT -v
  python compute_subject_level.py --subject sub-01 --condition Kill

For detailed usage, see TASKS.md or run: python compute_subject_level.py --help
"""

import os
import os.path as op
from typing import Optional, List
import pandas as pd
import warnings
import argparse
import logging
import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_sidecar_metadata, create_dataset_description
from nilearn.glm import cluster_level_inference
from nilearn.glm.second_level import SecondLevelModel
from nibabel import Nifti1Image

# Suppress informational warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="sub-06",
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-cond",
    "--condition",
    default="DOWNXlvl5",
    type=str,
    help="Condition (contrast) to process",
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
args = parser.parse_args()

def process_subject(
    sub: str,
    condition: str,
    path_to_data: str,
    logger: Optional[AnalysisLogger] = None
) -> Optional[Nifti1Image]:
    """
    Perform subject-level GLM analysis by combining session-level z-maps.

    Implements a random-effects analysis across sessions within a single subject,
    computing the mean activation and testing whether it differs significantly
    from zero across sessions.

    Args:
        sub: Subject identifier (e.g., 'sub-01')
        condition: Contrast/condition name (e.g., 'HIT', 'Kill', 'DOWNXlvl5')
        path_to_data: Root path to data directory
        logger: Optional logger instance for tracking progress

    Returns:
        Subject-level z-map as Nifti1Image, or None if processing failed

    Note:
        Input: Session-level z-maps from processed/session-level/
        Output: Subject-level maps in processed/subject-level/

        The analysis uses SecondLevelModel with an intercept-only design,
        equivalent to a one-sample t-test across sessions. Conservative
        cluster correction (Z > 3.1) is applied for family-wise error control.
    """
    # Read session-level z-maps from the new structure
    # We look for session-level (all runs) z-maps for this subject
    session_level_dir = op.join(path_to_data, "processed", "session-level", sub)

    if not op.exists(session_level_dir):
        msg = f"""
Session-level directory not found for {sub}: {session_level_dir}

Troubleshooting:
  1. Run session-level GLM analysis first: python shinobi_fmri/glm/compute_session_level.py -s {sub} -ses <session>
  2. Verify the subject ID '{sub}' is correct
  3. Check the data path: {path_to_data}
  4. Ensure session-level outputs were successfully generated

Expected directory: {session_level_dir}
"""
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    # Create output directories
    z_maps_out_dir = op.join(path_to_data, "processed", "subject-level", sub, "z_maps")
    beta_maps_out_dir = op.join(path_to_data, "processed", "subject-level", sub, "beta_maps")
    os.makedirs(z_maps_out_dir, exist_ok=True)
    os.makedirs(beta_maps_out_dir, exist_ok=True)

    # Collect all z-maps for this subject and condition
    z_map = None
    z_maps = []
    ses_list = []

    # Iterate through all sessions for this subject
    for ses_dir in sorted(os.listdir(session_level_dir)):
        ses_path = op.join(session_level_dir, ses_dir)
        if not op.isdir(ses_path):
            continue

        # Look for z-maps in the z_maps subdirectory
        z_maps_dir = op.join(ses_path, "z_maps")
        if not op.exists(z_maps_dir):
            continue

        # Find z-map files for this condition
        for file in os.listdir(z_maps_dir):
            if f"contrast-{condition}" in file and file.endswith("stat-z.nii.gz"):
                if logger:
                    logger.info(f"Adding : {file}")
                else:
                    print(f"Adding : {file}")
                z_maps.append(op.join(z_maps_dir, file))
                ses_list.append(ses_dir)

    if not z_maps:
        msg = f"""
No session-level z-maps found for {sub}, condition: {condition}

Searched in: {session_level_dir}

Troubleshooting:
  1. Run session-level GLM analysis first for this subject
  2. Verify the condition name '{condition}' is correct (case-sensitive)
  3. Check that session-level outputs exist in: {session_level_dir}/*/z_maps/
  4. Available sessions: {[d for d in os.listdir(session_level_dir) if op.isdir(op.join(session_level_dir, d))] if op.exists(session_level_dir) else 'directory not found'}

Expected filename pattern: *contrast-{condition}*stat-z.nii.gz
"""
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    # Output filename
    subjectlevel_z_map_fname = op.join(z_maps_out_dir, f"{sub}_task-shinobi_contrast-{condition}_stat-z.nii.gz")
    subjectlevel_beta_map_fname = op.join(beta_maps_out_dir, f"{sub}_task-shinobi_contrast-{condition}_stat-beta.nii.gz")

    if logger:
        logger.log_computation_start(f"SubjectLevel_{condition}", subjectlevel_z_map_fname)
    else:
        print(f"Computing subject level for {condition}")

    # Compute map
    try:
        second_level_input = z_maps
        second_design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])

        second_level_model = SecondLevelModel(smoothing_fwhm=None)
        second_level_model = second_level_model.fit(second_level_input,
                                                    design_matrix=second_design_matrix)

        z_map = second_level_model.compute_contrast(second_level_contrast=[1],
                                                    output_type='z_score',
                                                    second_level_stat_type="F")
        z_map.to_filename(subjectlevel_z_map_fname)

        if logger:
            logger.log_computation_success(f"SubjectLevel_{condition}", subjectlevel_z_map_fname)

        # Compute and save cluster-corrected Z-map (Conservative threshold)
        try:
            corrected_map = cluster_level_inference(z_map, threshold=config.GLM_CLUSTER_THRESH_SUBJECT, alpha=config.GLM_ALPHA)
            # BIDS-compliant naming: insert 'desc-corrected'
            corrected_fname = subjectlevel_z_map_fname.replace('_stat-z.nii.gz', '_desc-corrected_stat-z.nii.gz')
            corrected_map.to_filename(corrected_fname)
        except Exception as e:
            print(f"Warning: Failed to compute cluster correction for {condition}: {e}")

        # Compute beta map (using effect size instead of z-score)
        beta_map = second_level_model.compute_contrast(second_level_contrast=[1],
                                                       output_type='effect_size')
        beta_map.to_filename(subjectlevel_beta_map_fname)

        # Save metadata JSON sidecar for reproducibility
        metadata = create_metadata(
            description=f"Subject-level GLM z-map for contrast {condition}",
            script_path=__file__,
            output_files=[subjectlevel_z_map_fname, subjectlevel_beta_map_fname],
            parameters={
                'contrast': condition,
                'cluster_threshold': config.GLM_CLUSTER_THRESH_SUBJECT,
                'alpha': config.GLM_ALPHA,
                'smoothing_fwhm': None,
                'second_level_stat_type': 'F',
                'n_sessions': len(z_maps),
            },
            subject=sub,
            session=None,  # Subject-level combines all sessions
            additional_info={
                'analysis_level': 'subject',
                'sessions_included': ses_list,
                'input_z_maps': [op.basename(zm) for zm in z_maps],
            }
        )
        save_sidecar_metadata(subjectlevel_z_map_fname, metadata, logger=logger)

        # Create report
        report_path = op.join(config.FIG_PATH, "subject-level", condition, "report")
        os.makedirs(report_path, exist_ok=True)
        report_fname = op.join(report_path, f"{sub}_{condition}_report.html")
        report = second_level_model.generate_report(
            contrasts=['intercept'],
            height_control=None
        )
        report.save_as_html(report_fname)

    except Exception as e:
        if logger:
            logger.log_computation_error(f"SubjectLevel_{condition}", e)
        else:
            print(f"Error: {e}")

    return z_map


def main(logger=None):
    process_subject(sub, condition, path_to_data, logger=logger)

if __name__ == "__main__":
    figures_path = config.FIG_PATH
    path_to_data = config.DATA_PATH
    sub = args.subject
    condition = args.condition

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="GLM_subject",
        subject=sub,
        condition=condition,
        log_dir=args.log_dir,
        verbosity=log_level
    )
    
    logger.info(f"Processing : {sub} {condition}")
    logger.info(f"Writing processed data in : {path_to_data}")
    logger.info(f"Writing reports in : {figures_path}")

    # Create dataset_description.json for processed outputs
    dataset_desc_dir = op.join(path_to_data, "processed", "subject-level")
    if not op.exists(op.join(dataset_desc_dir, "dataset_description.json")):
        create_dataset_description(
            name="Subject-level GLM Analysis",
            description="Third-level GLM analysis combining session-level z-maps with random-effects",
            pipeline_version="0.1.0",
            derived_from="Session-level GLM z-maps",
            parameters={
                'cluster_threshold': config.GLM_CLUSTER_THRESH_SUBJECT,
                'alpha': config.GLM_ALPHA,
                'smoothing_fwhm': None,
                'model_type': 'SecondLevelModel',
            },
            output_dir=dataset_desc_dir
        )
        logger.info(f"Created dataset_description.json in {dataset_desc_dir}")

    try:
        main(logger=logger)
    finally:
        logger.close()