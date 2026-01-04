#!/usr/bin/env python
"""
Apply cluster-level FWE correction to existing z-maps.

This script takes raw uncorrected z-maps and applies cluster-level family-wise
error (FWE) correction, saving properly thresholded z-maps where only voxels
in significant clusters retain their original z-values.

Unlike the original cluster_level_inference output (which returns p-value maps),
this script saves actual z-score maps for easy visualization.

Statistical Method:
  - Uses nilearn's cluster_level_inference for FWE correction
  - Cluster-forming threshold: configurable (default from config)
  - Family-wise error rate: alpha (default: 0.05)
  - Only clusters with p < alpha survive correction

Output:
  - Thresholded z-maps: *_desc-corrected_stat-z.nii.gz
  - Contains original z-values for voxels in FWE-significant clusters
  - Zero elsewhere

Usage:
    python apply_cluster_correction.py --level subject
    python apply_cluster_correction.py --level session --subject sub-01
    python apply_cluster_correction.py --level session --subject sub-01 --session ses-001
    python apply_cluster_correction.py --threshold 2.3 --alpha 0.05 --overwrite
"""

import os
import os.path as op
import argparse
import glob
import logging
import numpy as np
import nibabel as nib
from nilearn.glm import cluster_level_inference
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
    Apply cluster-level FWE correction to a single z-map.

    Args:
        raw_zmap_path: Path to raw uncorrected z-map
        corrected_zmap_path: Path to save corrected z-map
        threshold: Cluster-forming threshold (z-score)
        alpha: Family-wise error rate for cluster correction
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
        z_data = z_map.get_fdata()

        # Apply cluster-level inference
        # This returns a p-value map where:
        #   - 0 = voxel not in any cluster
        #   - >0 = cluster-level FWE-corrected p-value
        p_map = cluster_level_inference(z_map, threshold=threshold, alpha=alpha)
        p_data = p_map.get_fdata()

        # Create thresholded z-map
        # Keep original z-values only for voxels in significant clusters (p < alpha)
        sig_mask = (p_data > 0) & (p_data < alpha)
        thresholded_z = np.zeros_like(z_data)
        thresholded_z[sig_mask] = z_data[sig_mask]

        # Save thresholded z-map
        corrected_img = nib.Nifti1Image(thresholded_z, z_map.affine, z_map.header)
        corrected_img.to_filename(corrected_zmap_path)

        # Log statistics
        n_sig_voxels = np.sum(sig_mask)
        if logger:
            if n_sig_voxels > 0:
                max_z = np.max(np.abs(thresholded_z))
                logger.info(f"✓ {op.basename(corrected_zmap_path)}: {n_sig_voxels} voxels survive (max |z|={max_z:.2f})")
            else:
                # Get minimum cluster p-value to report
                cluster_ps = p_data[p_data > 0]
                min_p = np.min(cluster_ps) if len(cluster_ps) > 0 else 1.0
                logger.warning(f"✗ {op.basename(corrected_zmap_path)}: No significant clusters (min p={min_p:.3f})")

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
        logger: Logger instance
    """
    if threshold is None:
        threshold = config.GLM_CLUSTER_THRESH_SUBJECT

    # Get subjects to process
    subjects = [subject] if subject else config.SUBJECTS

    if logger:
        logger.info(f"Processing subject-level z-maps (threshold={threshold}, alpha={alpha})")

    total_processed = 0
    total_success = 0

    for subj in subjects:
        zmaps_dir = op.join(data_path, "processed", "subject-level", subj, "z_maps")
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
        logger: Logger instance
    """
    if threshold is None:
        threshold = config.GLM_CLUSTER_THRESH_SESSION

    # Get subjects to process
    subjects = [subject] if subject else config.SUBJECTS

    if logger:
        logger.info(f"Processing session-level z-maps (threshold={threshold}, alpha={alpha})")

    total_processed = 0
    total_success = 0

    for subj in subjects:
        subj_dir = op.join(data_path, "processed", "session-level", subj)
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
        logger.info("CLUSTER-LEVEL FWE CORRECTION")
        logger.info("="*70)
        logger.info(f"Level: {args.level}")
        logger.info(f"Alpha: {args.alpha}")

        # Display thresholds
        if args.level in ["subject", "both"]:
            thresh_subj = args.threshold if args.threshold else config.GLM_CLUSTER_THRESH_SUBJECT
            logger.info(f"Subject-level threshold: {thresh_subj}")

        if args.level in ["session", "both"]:
            thresh_sess = args.threshold if args.threshold else config.GLM_CLUSTER_THRESH_SESSION
            logger.info(f"Session-level threshold: {thresh_sess}")

        logger.info("="*70)

        # Process requested levels
        if args.level in ["subject", "both"]:
            process_subject_level(
                args.data_path,
                args.subject,
                args.threshold,
                args.alpha,
                args.overwrite,
                logger
            )

        if args.level in ["session", "both"]:
            process_session_level(
                args.data_path,
                args.subject,
                args.session,
                args.threshold,
                args.alpha,
                args.overwrite,
                logger
            )

        logger.info("\n" + "="*70)
        logger.info("COMPLETE")
        logger.info("="*70)

    finally:
        logger.close()


if __name__ == "__main__":
    main()
