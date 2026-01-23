#!/usr/bin/env python
"""
Compare corrected vs uncorrected z-maps using raw volume slices.

This script creates orthogonal slice plots showing raw voxel data from both
corrected and uncorrected z-maps side-by-side, with no interpolation.

Usage:
    # Compare specific subject and condition
    python viz_volume_slices.py --subject sub-01 --condition HIT

    # Specify custom coordinates for slicing
    python viz_volume_slices.py --subject sub-01 --condition Kill --coords 45,55,35

    # Session-level comparison
    python viz_volume_slices.py --subject sub-01 --session ses-004 --condition HIT
"""

import os
import os.path as op
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from shinobi_fmri.utils.logger import AnalysisLogger
import logging

try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS
except ImportError:
    print("Warning: config not found. Using defaults.")
    DATA_PATH = "/home/hyruuk/scratch/data"
    SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']


def compare_volume_slices(uncorr_path, corr_path, coords=None, save_path=None, logger=None):
    """Create side-by-side comparison of uncorrected and corrected z-maps.

    Args:
        uncorr_path (str): Path to uncorrected z-map
        corr_path (str): Path to corrected z-map
        coords (tuple): (x, y, z) coordinates for slicing. If None, uses center of mass
        save_path (str): Path to save figure
        logger: Logger instance
    """
    # Load maps
    uncorr_img = nib.load(uncorr_path)
    uncorr_data = uncorr_img.get_fdata()

    if op.isfile(corr_path):
        corr_img = nib.load(corr_path)
        corr_data = corr_img.get_fdata()
    else:
        if logger:
            logger.warning(f"Corrected map not found: {corr_path}")
        corr_img = None
        corr_data = None

    # Get coordinate for slicing
    if coords is None:
        # Use center of mass of uncorrected data
        from scipy import ndimage
        com = ndimage.center_of_mass(np.abs(uncorr_data) > 3)
        coords = tuple(int(c) for c in com)
        if logger:
            logger.info(f"Using center of mass coordinates: {coords}")

    # Determine value range for consistent colormap
    if corr_data is not None:
        vmax = max(np.abs(uncorr_data).max(), np.abs(corr_data).max())
    else:
        vmax = np.abs(uncorr_data).max()
    vmin = -vmax

    if logger:
        logger.info(f"Value range: [{vmin:.2f}, {vmax:.2f}]")
        logger.info(f"Uncorrected - non-zero voxels: {np.sum(uncorr_data != 0)}")
        if corr_data is not None:
            logger.info(f"Corrected - non-zero voxels: {np.sum(corr_data != 0)}")

    # Create figure
    n_rows = 2 if corr_data is not None else 1
    fig = plt.figure(figsize=(15, 5 * n_rows))

    # Plot uncorrected map
    # Use small threshold to hide voxels that are exactly zero or very close
    display = plotting.plot_stat_map(
        uncorr_img,
        cut_coords=coords,
        display_mode='ortho',
        threshold=0.01,  # Hide voxels very close to zero
        vmax=vmax,
        title=f"Uncorrected z-map\n{op.basename(uncorr_path)}",
        figure=fig,
        axes=(0, 0.5, 1, 0.5) if n_rows == 2 else None,
        colorbar=True,
        annotate=True,
        draw_cross=True
    )

    # Plot corrected map
    if corr_data is not None:
        # Use same small threshold to hide zeros (most of corrected map is zero)
        display = plotting.plot_stat_map(
            corr_img,
            cut_coords=coords,
            display_mode='ortho',
            threshold=0.01,  # Hide voxels very close to zero
            vmax=vmax,
            title=f"Corrected z-map\n{op.basename(corr_path)}",
            figure=fig,
            axes=(0, 0, 1, 0.5),
            colorbar=True,
            annotate=True,
            draw_cross=True
        )

    plt.suptitle(f"Volume Slice Comparison at coords {coords}", fontsize=14, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if logger:
            logger.info(f"Saved: {save_path}")

    plt.close(fig)

    return coords


def plot_value_histograms(uncorr_path, corr_path, save_path=None, logger=None):
    """Plot histograms of voxel values for uncorrected and corrected maps.

    Args:
        uncorr_path (str): Path to uncorrected z-map
        corr_path (str): Path to corrected z-map
        save_path (str): Path to save figure
        logger: Logger instance
    """
    # Load maps
    uncorr_data = nib.load(uncorr_path).get_fdata()

    if op.isfile(corr_path):
        corr_data = nib.load(corr_path).get_fdata()
    else:
        corr_data = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Uncorrected - all values
    ax = axes[0, 0]
    uncorr_nonzero = uncorr_data[uncorr_data != 0]
    ax.hist(uncorr_nonzero, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Uncorrected (all non-zero)\nn={len(uncorr_nonzero)}')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)

    # Uncorrected - absolute values
    ax = axes[0, 1]
    ax.hist(np.abs(uncorr_nonzero), bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('|Z-score|')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncorrected (absolute values)')
    ax.axvline(3, color='red', linestyle='--', linewidth=1, label='|z|=3')
    ax.legend()
    ax.grid(alpha=0.3)

    if corr_data is not None:
        # Corrected - all values
        ax = axes[1, 0]
        corr_nonzero = corr_data[corr_data != 0]
        ax.hist(corr_nonzero, bins=100, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Z-score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Corrected (all non-zero)\nn={len(corr_nonzero)}')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.grid(alpha=0.3)

        # Corrected - absolute values
        ax = axes[1, 1]
        ax.hist(np.abs(corr_nonzero), bins=100, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('|Z-score|')
        ax.set_ylabel('Frequency')
        ax.set_title('Corrected (absolute values)')
        ax.axvline(3, color='red', linestyle='--', linewidth=1, label='|z|=3')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Corrected map not found',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Corrected map not found',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

    plt.suptitle(f'Value Distribution Comparison\n{op.basename(uncorr_path)}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if logger:
            logger.info(f"Saved: {save_path}")

    plt.close(fig)


def main():
    """Main function to generate volume slice comparisons."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Subject ID (e.g., sub-01)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        help='Condition/contrast name (e.g., HIT, Kill)'
    )
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Session ID for session-level maps (e.g., ses-004). If not provided, uses subject-level.'
    )
    parser.add_argument(
        '--coords',
        type=str,
        default=None,
        help='Slice coordinates as x,y,z (e.g., "45,55,35"). If not provided, uses center of mass.'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DATA_PATH,
        help=f'Path to data directory (default: {DATA_PATH})'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: ./reports/figures/volume_slices/)'
    )
    parser.add_argument(
        '--exclude-low-level',
        action='store_true',
        help='Exclude low-level features from visualization (default: False, low-level features included)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action="count",
        default=0,
        help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Parse coordinates
    coords = None
    if args.coords:
        try:
            coords = tuple(int(c) for c in args.coords.split(','))
            if len(coords) != 3:
                raise ValueError("Coordinates must be x,y,z")
        except Exception as e:
            parser.error(f"Invalid coordinates: {e}")

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="VolumeSlices",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        # Always use processed directory (low-level features are now default)
        output_dir = "processed"

        if args.output_dir is None:
            args.output_dir = op.join(".", "reports", "figures", "volume_slices")
        os.makedirs(args.output_dir, exist_ok=True)

        # Determine level (subject or session)
        if args.session:
            level = "session-level"
            level_id = f"{args.subject}_{args.session}"
            base_path = op.join(
                args.data_path, output_dir, level, args.subject, args.session, "z_maps"
            )
            uncorr_filename = f"{args.subject}_{args.session}_task-shinobi_contrast-{args.condition}_stat-z.nii.gz"
            corr_filename = f"{args.subject}_{args.session}_task-shinobi_contrast-{args.condition}_desc-corrected_stat-z.nii.gz"
        else:
            level = "subject-level"
            level_id = args.subject
            base_path = op.join(
                args.data_path, output_dir, level, args.subject, "z_maps"
            )
            uncorr_filename = f"{args.subject}_task-shinobi_contrast-{args.condition}_stat-z.nii.gz"
            corr_filename = f"{args.subject}_task-shinobi_contrast-{args.condition}_desc-corrected_stat-z.nii.gz"

        # Full paths
        uncorr_path = op.join(base_path, uncorr_filename)
        corr_path = op.join(base_path, corr_filename)

        logger.info(f"Comparing {level} maps for {level_id}, condition: {args.condition}")
        logger.info(f"Uncorrected: {uncorr_path}")
        logger.info(f"Corrected: {corr_path}")

        # Check if uncorrected map exists
        if not op.isfile(uncorr_path):
            logger.error(f"Uncorrected map not found: {uncorr_path}")
            return

        # Generate volume slice comparison
        slice_save_path = op.join(
            args.output_dir,
            f"{level_id}_{args.condition}_volume_slices.png"
        )
        final_coords = compare_volume_slices(
            uncorr_path, corr_path, coords, slice_save_path, logger
        )

        # Generate histogram comparison
        hist_save_path = op.join(
            args.output_dir,
            f"{level_id}_{args.condition}_histograms.png"
        )
        plot_value_histograms(uncorr_path, corr_path, hist_save_path, logger)

        logger.info("\nAll done!")
        logger.info(f"Slice comparison saved to: {op.abspath(slice_save_path)}")
        logger.info(f"Histogram comparison saved to: {op.abspath(hist_save_path)}")
        logger.info(f"Slice coordinates used: {final_coords}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
