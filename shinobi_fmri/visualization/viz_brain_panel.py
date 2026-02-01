#!/usr/bin/env python3
"""
Composite Figure Panel Visualization

Generates a composite figure panel (A, B, C) from three input images.

Default images (Kill condition analysis):
- Panel A: Kill condition annotations (subject-level brain maps)
- Panel B: Kill vs audio envelope comparison
- Panel C: Kill vs reward comparison

Output is sized to approximately 3/4 of an A4 page.

Usage:
    python viz_figure_panel.py [-v] [--output path/to/figure.png]
    python viz_figure_panel.py --panel-a img1.png --panel-b img2.png --panel-c img3.png
    invoke viz.figure-panel
"""
import os
import os.path as op
import argparse
import logging

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger


# Figure dimensions: 3/4 of A4 page at 300 DPI
# A4 = 210 x 297 mm, 3/4 height = ~223 mm
# At 300 DPI: 210mm = 8.27in, 223mm = 8.78in
FIGURE_WIDTH_INCHES = 8.27
FIGURE_HEIGHT_INCHES = 8.78
DPI = 300


def load_image(path: str, logger: logging.Logger) -> any:
    """
    Load an image file.

    Args:
        path: Path to the image file
        logger: Logger instance

    Returns:
        Image array

    Raises:
        FileNotFoundError: If the image file does not exist
    """
    if not op.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    logger.debug(f"Loading image: {path}")
    return mpimg.imread(path)


def create_figure_panel(
    panel_a_path: str,
    panel_b_path: str,
    panel_c_path: str,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Create composite figure panel with A, B, C layout.

    Layout:
    - Panel A spans the top (full width)
    - Panels B and C side by side at the bottom

    Args:
        panel_a_path: Path to panel A image
        panel_b_path: Path to panel B image
        panel_c_path: Path to panel C image
        output_path: Path to save the output figure
        logger: Logger instance
    """
    logger.info("Creating composite figure panel")

    # Load images
    img_a = load_image(panel_a_path, logger)
    img_b = load_image(panel_b_path, logger)
    img_c = load_image(panel_c_path, logger)

    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=DPI)

    # Create grid: 2 rows, 2 columns
    # Row 0: Panel A spans both columns
    # Row 1: Panel B (left), Panel C (right)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.2],  # Bottom row slightly taller for B and C
        width_ratios=[1, 1],
        hspace=0.02,  # Minimal vertical space between A and B/C
        wspace=0.02,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.32,  # Push B and C up by ~30%
    )

    # Panel A - spans top row
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.imshow(img_a)
    ax_a.axis('off')
    ax_a.text(
        0.01, 0.95, 'A',
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
    )

    # Panel B - bottom left
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.imshow(img_b)
    ax_b.axis('off')
    ax_b.text(
        0.01, 1.0, 'B',
        transform=ax_b.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='bottom',
    )

    # Panel C - bottom right
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.imshow(img_c)
    ax_c.axis('off')
    ax_c.text(
        0.01, 1.0, 'C',
        transform=ax_c.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='bottom',
    )

    # Save figure
    os.makedirs(op.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Figure saved: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate composite figure panel (A, B, C) from three images"
    )
    parser.add_argument(
        '--panel-a',
        type=str,
        default=None,
        help='Path to panel A image (default: annotations_plot_Kill_corrected.png)',
    )
    parser.add_argument(
        '--panel-b',
        type=str,
        default=None,
        help='Path to panel B image (default: Kill_vs_audio_envelope_corrected_panel.png)',
    )
    parser.add_argument(
        '--panel-c',
        type=str,
        default=None,
        help='Path to panel C image (default: Kill_vs_reward_corrected_panel.png)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output figure path (default: {FIG_PATH}/brain_panels.png)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (use -v for INFO, -vv for DEBUG)',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Custom log directory',
    )
    return parser.parse_args()


def main():
    """Main function to generate composite figure panel."""
    args = parse_args()

    # Setup logging
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO

    log_dir = args.log_dir or 'logs/visualization'
    analysis_logger = AnalysisLogger(
        log_name='viz_figure_panel',
        log_dir=log_dir,
        verbosity=log_level,
    )
    logger = analysis_logger.logger

    # Default paths (Kill condition images)
    fig_path = config.FIG_PATH

    panel_a_path = args.panel_a or op.join(
        fig_path, 'full_zmap_plot', 'annotations',
        'annotations_plot_Kill_corrected.png'
    )
    panel_b_path = args.panel_b or op.join(
        fig_path, 'condition_comparison',
        'Kill_vs_audio_envelope_corrected_panel.png'
    )
    panel_c_path = args.panel_c or op.join(
        fig_path, 'condition_comparison',
        'Kill_vs_reward_corrected_panel.png'
    )
    output_path = args.output or op.join(fig_path, 'brain_panels.png')

    logger.info("Composite Figure Panel Generation")
    logger.info(f"  Panel A: {panel_a_path}")
    logger.info(f"  Panel B: {panel_b_path}")
    logger.info(f"  Panel C: {panel_c_path}")
    logger.info(f"  Output: {output_path}")

    # Create figure panel
    create_figure_panel(
        panel_a_path=panel_a_path,
        panel_b_path=panel_b_path,
        panel_c_path=panel_c_path,
        output_path=output_path,
        logger=logger,
    )

    print(f"Figure panel saved: {output_path}")


if __name__ == '__main__':
    main()
