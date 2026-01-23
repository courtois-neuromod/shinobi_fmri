#!/usr/bin/env python
"""
Apply FDR correction to existing z-maps.

This script takes raw uncorrected z-maps and applies voxel-wise False Discovery
Rate (FDR) correction, saving properly thresholded z-maps where only significant
voxels retain their original z-values.

Statistical Method:
  - Uses nilearn's threshold_stats_img for FDR correction
  - Height control: FDR
  - Alpha: False Discovery Rate (default: 0.05)
  - Returns actual Z-threshold used

Output:
  - Thresholded z-maps: *_desc-corrected_stat-z.nii.gz
  - Contains original z-values for voxels significant at FDR < alpha
  - Zero elsewhere

Usage:
    python apply_cluster_correction.py --level subject
    python apply_cluster_correction.py --level session --subject sub-01
    python apply_cluster_correction.py --alpha 0.05 --overwrite
"""

import os
import os.path as op
import argparse
import glob
import logging
import numpy as np
import nibabel as nib
from nilearn.glm import threshold_stats_img
from tqdm import tqdm
from shinobi_fmri import config
from shinobi_fmri.utils.logger import AnalysisLogger


def apply_cluster_correction_to_map(
    raw_zmap_path,
    corrected_zmap_path,
    threshold,
    alpha,
    overwrite=False,
    logger=None
):
    """
    Apply FDR correction to a single z-map.

    Args:
        raw_zmap_path: Path to raw uncorrected z-map
        corrected_zmap_path: Path to save corrected z-map
        threshold: Ignored for FDR (kept for API compatibility)
        alpha: False Discovery Rate (FDR) q-value (e.g. 0.05)
        overwrite: Overwrite existing corrected maps
        logger: Logger instance

    Returns:
        bool: True if successful, False otherwise
    """
    # Skip if output exists and not overwriting
    if op.exists(corrected_zmap_path) and not overwrite:
        if logger:
            logger.debug(f"Skipping (exists): {op.basename(corrected_zmap_path)}")
        return True

    # Check input exists
    if not op.exists(raw_zmap_path):
        if logger:
            logger.warning(f"Input not found: {raw_zmap_path}")
        return False

    try:
        # Load raw z-map
        z_map = nib.load(raw_zmap_path)

        # Apply FDR correction
        # threshold_stats_img returns the thresholded image and the threshold value used
        thresholded_z_img, threshold_value = threshold_stats_img(
            z_map,
            alpha=alpha,
            height_control='fdr',
            cluster_threshold=10,
        )
        
        # Save thresholded z-map
        thresholded_z_img.to_filename(corrected_zmap_path)

        # Log statistics
        # Check non-zero voxels
        thresholded_data = thresholded_z_img.get_fdata()
        n_sig_voxels = np.count_nonzero(thresholded_data)
        
        if logger:
            if n_sig_voxels > 0:
                max_z = np.max(np.abs(thresholded_data))
                logger.info(f"✓ {op.basename(corrected_zmap_path)}: {n_sig_voxels} voxels survive FDR q<{alpha} (Z > {threshold_value:.2f}, max |z|={max_z:.2f})")
            else:
                logger.warning(f"✗ {op.basename(corrected_zmap_path)}: No voxels survive FDR q<{alpha}")

        return True

    except Exception as e:
        if logger:
            logger.error(f"Error processing {raw_zmap_path}: {e}")
        return False


def process_subject_level(
    data_path,
    subject=None,
    threshold=None,
    alpha=0.05,
    overwrite=False,
    low_level_confs=True,
    logger=None
):
    """
    Apply cluster correction to subject-level z-maps.

    Args:
        data_path: Root data directory
        subject: Subject ID (process all if None)
        threshold: Cluster-forming threshold (use config default if None)
        alpha: FWE alpha level
        overwrite: Overwrite existing files
        low_level_confs: Use results from GLM with low-level confounds
        logger: Logger instance
    """
    if threshold is None:
        threshold = config.GLM_CLUSTER_THRESH_SUBJECT

    # Get subjects to process
    subjects = [subject] if subject else config.SUBJECTS

    # Always use processed directory (low-level features are now default)
    processed_dir = "processed"

    if logger:
        logger.info(f"Processing subject-level z-maps (threshold={threshold}, alpha={alpha})")
        logger.info(f"Directory: {processed_dir}")

    total_processed = 0
    total_success = 0

    for subj in subjects:
        zmaps_dir = op.join(data_path, processed_dir, "subject-level", subj, "z_maps")
        if not op.exists(zmaps_dir):
            if logger:
                logger.warning(f"Directory not found: {zmaps_dir}")
            continue

        # Find all raw z-maps (exclude already corrected ones)
        raw_zmaps = glob.glob(op.join(zmaps_dir, "*_stat-z.nii.gz"))
        raw_zmaps = [z for z in raw_zmaps if "desc-corrected" not in z]

        if logger:
            logger.info(f"Processing {subj}: {len(raw_zmaps)} z-maps")

        for raw_path in tqdm(raw_zmaps, desc=f"  {subj}", disable=logger is None):
            total_processed += 1
            corrected_path = raw_path.replace("_stat-z.nii.gz", "_desc-corrected_stat-z.nii.gz")

            success = apply_cluster_correction_to_map(
                raw_path, corrected_path, threshold, alpha, overwrite, logger
            )
            if success:
                total_success += 1

    if logger:
        logger.info(f"Subject-level: {total_success}/{total_processed} z-maps processed successfully")


def process_session_level(
    data_path,
    subject=None,
    session=None,
    threshold=None,
    alpha=0.05,
    overwrite=False,
    low_level_confs=True,
    logger=None
):
    """
    Apply cluster correction to session-level z-maps.

    Args:
        data_path: Root data directory
        subject: Subject ID (process all if None)
        session: Session ID (process all if None)
        threshold: Cluster-forming threshold (use config default if None)
        alpha: FWE alpha level
        overwrite: Overwrite existing files
        low_level_confs: Use results from GLM with low-level confounds
        logger: Logger instance
    """
    if threshold is None:
        threshold = config.GLM_CLUSTER_THRESH_SESSION

    # Get subjects to process
    subjects = [subject] if subject else config.SUBJECTS

    # Always use processed directory (low-level features are now default)
    processed_dir = "processed"

    if logger:
        logger.info(f"Processing session-level z-maps (threshold={threshold}, alpha={alpha})")
        logger.info(f"Directory: {processed_dir}")

    total_processed = 0
    total_success = 0

    for subj in subjects:
        subj_dir = op.join(data_path, processed_dir, "session-level", subj)
        if not op.exists(subj_dir):
            if logger:
                logger.warning(f"Directory not found: {subj_dir}")
            continue

        # Get sessions
        if session:
            sessions = [session]
        else:
            sessions = [d for d in os.listdir(subj_dir) if d.startswith("ses-") and op.isdir(op.join(subj_dir, d))]
            sessions = sorted(sessions)

        for sess in sessions:
            zmaps_dir = op.join(subj_dir, sess, "z_maps")
            if not op.exists(zmaps_dir):
                continue

            # Find all raw z-maps
            raw_zmaps = glob.glob(op.join(zmaps_dir, "*_stat-z.nii.gz"))
            raw_zmaps = [z for z in raw_zmaps if "desc-corrected" not in z]

            if logger and len(raw_zmaps) > 0:
                logger.info(f"Processing {subj}/{sess}: {len(raw_zmaps)} z-maps")

            for raw_path in tqdm(raw_zmaps, desc=f"  {subj}/{sess}", disable=logger is None):
                total_processed += 1
                corrected_path = raw_path.replace("_stat-z.nii.gz", "_desc-corrected_stat-z.nii.gz")

                success = apply_cluster_correction_to_map(
                    raw_path, corrected_path, threshold, alpha, overwrite, logger
                )
                if success:
                    total_success += 1

    if logger:
        logger.info(f"Session-level: {total_success}/{total_processed} z-maps processed successfully")


def main():
    """Main function to apply cluster correction to z-maps."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["subject", "session", "both"],
        help="Analysis level to process"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process specific subject only (default: all subjects)"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process specific session only (session-level only, default: all sessions)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cluster-forming threshold (default: from config)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise error rate (default: 0.05)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=config.DATA_PATH,
        help=f"Path to data directory (default: {config.DATA_PATH})"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corrected z-maps"
    )
    # Removed --low-level-confs flag - low-level features are now default
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files"
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
    logger = AnalysisLogger(
        log_name="ClusterCorrection",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        logger.info("="*70)
        logger.info("FDR CORRECTION")
        logger.info("="*70)
        logger.info(f"Level: {args.level}")
        logger.info(f"Alpha (FDR q): {args.alpha}")

        # Low-level features are now always included by default
        logger.info("Using processed/ directory (low-level features included by default)")

        # Display thresholds (Ignored for FDR but logged for record if passed)
        if args.threshold:
             logger.info(f"Note: --threshold {args.threshold} is ignored for FDR calculation.")

        logger.info("="*70)

        # Process requested levels
        if args.level in ["subject", "both"]:
            process_subject_level(
                args.data_path,
                args.subject,
                args.threshold,
                args.alpha,
                args.overwrite,
                low_level_confs=True,
                logger=logger
            )

        if args.level in ["session", "both"]:
            process_session_level(
                args.data_path,
                args.subject,
                args.session,
                args.threshold,
                args.alpha,
                args.overwrite,
                low_level_confs=True,
                logger=logger
            )

        logger.info("\n" + "="*70)
        logger.info("COMPLETE")
        logger.info("="*70)

    finally:
        logger.close()


if __name__ == "__main__":
    main()
