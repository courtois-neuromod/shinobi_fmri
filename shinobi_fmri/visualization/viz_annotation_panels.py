#!/usr/bin/env python
"""
Generate annotation panels showing subject-level and session-level brain maps.

This script creates:
1. Individual inflated brain maps for subject-level and session-level z-maps
2. Combined panel figures showing subject-level map + top 4 session-level maps per subject
3. A PDF containing panels for all annotations

For each annotation/condition, creates a 4x9 grid showing:
- 4 subjects (in 2x2 layout)
- For each subject: 1 large subject-level map + 4 smaller session-level maps
- The 4 session-level maps are selected based on number of voxels above threshold

Usage:
    python viz_annotation_panels.py                                    # Process all conditions
    python viz_annotation_panels.py --condition HIT                    # Process single condition
    python viz_annotation_panels.py --conditions HIT,JUMP,Kill         # Process multiple conditions
    python viz_annotation_panels.py --output-dir ./custom_output       # Custom output directory
"""

import os
import os.path as op
import argparse
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import nibabel as nib
from nilearn import plotting
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape

try:
    import shinobi_behav
    DATA_PATH = shinobi_behav.DATA_PATH
    SUBJECTS = shinobi_behav.SUBJECTS
except ImportError:
    print("Warning: shinobi_behav not found. Using defaults.")
    DATA_PATH = "/home/hyruuk/scratch/data"
    SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']


# Default conditions to process
DEFAULT_CONDITIONS = ['Kill', 'HealthLoss', 'JUMP', 'HIT', 'DOWN', 'LEFT', 'RIGHT', 'UP']


def create_colormap():
    """Create custom colormap with grey zone for low values (-3 to 3).

    Returns:
        tuple: (fig, ax, mappable) - Figure, axis, and scalar mappable for colorbar
    """
    # Get the nilearn cold_hot colormap
    cmap = nilearn_cmaps['cold_hot']

    # Create a list of colors from the colormap
    colors = cmap(np.linspace(0, 1, cmap.N))

    # Set middle range (corresponding to -3 to 3) to grey
    lower_bound = int(64)  # -3 corresponds to this index
    upper_bound = int(192)  # 3 corresponds to this index
    colors[lower_bound:upper_bound, :] = [0.5, 0.5, 0.5, 1]  # RGBA for grey

    # Create new colormap with modified colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list('new_cmap', colors)

    # Create normalization for data range (-6, 6)
    norm = mcolors.Normalize(vmin=-6, vmax=6)

    # Create ScalarMappable for colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=new_cmap)
    mappable.set_array([])

    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(1, 6))
    plt.colorbar(mappable, ax=ax, orientation='vertical')
    ax.remove()

    return fig, ax, mappable


def plot_inflated_zmap(img, save_path=None, title=None, colorbar=True, vmax=6, threshold=3, dpi=300):
    """Plot inflated brain surface map and save the image.

    Args:
        img (str or Nifti1Image): Path to image or nibabel image object
        save_path (str): Path to save the image
        title (str): Title of the image
        colorbar (bool): Whether to include a colorbar
        vmax (float): Maximum value for the colormap
        threshold (float): Threshold value for displaying voxels
        dpi (int): DPI of the output image
    """
    # Set font size based on title
    if 'sub' in title:
        fontsize = 24
    elif 'ses' in title:
        fontsize = 32
    else:
        fontsize = 20

    plt.rcParams['figure.dpi'] = dpi

    # Plot on inflated surface
    plotting.plot_img_on_surf(
        img,
        views=["lateral", "medial"],
        hemispheres=["left", "right"],
        inflate=True,
        colorbar=colorbar,
        threshold=threshold,
        vmax=vmax,
        symmetric_cbar=False,
        darkness=None  # Suppress deprecation warning
    )

    # Get current figure and add title
    fig = plt.gcf()
    fig.canvas.draw()
    fig.suptitle(title, fontsize=fontsize, y=1.05)

    # Save with tight layout
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def create_all_images(subject, condition, fig_folder):
    """Create all individual brain map images for a subject and condition.

    This includes:
    - One subject-level z-map
    - Multiple session-level z-maps (one per session)

    Args:
        subject (str): Subject ID (e.g., 'sub-01')
        condition (str): Condition/annotation name
        fig_folder (str): Path to save the images
    """
    print(f"Creating images for {subject} {condition}")

    # Create subject-level z-map
    sublevel_zmap_path = op.join(
        DATA_PATH,
        "processed",
        "z_maps",
        "subject-level",
        condition,
        f"{subject}_simplemodel_{condition}.nii.gz"
    )
    sublevel_save_path = op.join(fig_folder, f"{subject}_{condition}.png")

    if not op.isfile(sublevel_save_path):
        if op.isfile(sublevel_zmap_path):
            plot_inflated_zmap(
                sublevel_zmap_path,
                save_path=sublevel_save_path,
                title=f"{subject}",
                colorbar=False
            )
        else:
            # Create blank image if z-map doesn't exist
            print(f"Warning: {sublevel_zmap_path} not found, creating blank image")
            create_blank_image(sublevel_save_path, fig_folder)

    # Create session-level z-maps
    ses_dir = op.join(DATA_PATH, "shinobi", subject)
    if op.exists(ses_dir):
        ses_list = sorted(os.listdir(ses_dir))

        for session in ses_list:
            zmap_path = op.join(
                DATA_PATH,
                "processed",
                "z_maps",
                "ses-level",
                condition,
                f"{subject}_{session}_simplemodel_{condition}.nii.gz"
            )
            save_path = op.join(fig_folder, f"{subject}_{session}_{condition}.png")

            if not op.isfile(save_path):
                if op.isfile(zmap_path):
                    plot_inflated_zmap(
                        zmap_path,
                        save_path=save_path,
                        title=f"{session}",
                        colorbar=False
                    )
                else:
                    # Create blank image if z-map doesn't exist
                    print(f"Warning: {zmap_path} not found, creating blank image")
                    create_blank_image(save_path, fig_folder)


def create_blank_image(save_path, fig_folder):
    """Create a blank white image matching size of other images.

    Args:
        save_path (str): Path to save the blank image
        fig_folder (str): Folder containing other images to match size
    """
    # Try to find an existing image to match size
    other_images = glob.glob(op.join(fig_folder, "*.png"))

    if other_images:
        img_size = Image.open(other_images[0]).size
    else:
        # Default size if no other images exist
        img_size = (2000, 1000)

    missing_img = Image.new('RGB', img_size, color=(255, 255, 255))
    missing_img.save(save_path)


def make_annotation_plot(condition, save_path):
    """Create a combined panel showing all subjects for one annotation.

    Creates a 4x9 grid showing:
    - 4 subjects (in 2x2 arrangement)
    - For each subject: 1 subject-level map + top 4 session-level maps
    - Colorbar on the right

    Args:
        condition (str): Condition/annotation name
        save_path (str): Path to save the combined panel
    """
    print(f"\nCreating annotation plot for {condition}")

    images = []

    for idx_subj, subject in enumerate(SUBJECTS):
        print(f"  Processing {subject}")
        fig_folder = op.join(".", "reports", "figures", "full_zmap_plot", subject, condition)

        # Find top 4 session maps based on number of voxels above threshold
        ses_dir = op.join(DATA_PATH, "shinobi", subject)
        if not op.exists(ses_dir):
            print(f"    Warning: Session directory not found for {subject}")
            continue

        ses_list = sorted(os.listdir(ses_dir))
        sesmap_vox_above_thresh = []
        sesmap_name = []

        for session in ses_list:
            try:
                zmap_path = op.join(
                    DATA_PATH,
                    "processed",
                    "z_maps",
                    "ses-level",
                    condition,
                    f"{subject}_{session}_simplemodel_{condition}.nii.gz"
                )

                if op.isfile(zmap_path):
                    img = nib.load(zmap_path).get_fdata()
                    # Count voxels with |z| > 3
                    nb_vox_above_thresh = np.sum(np.abs(img.flatten()) > 3)
                    sesmap_vox_above_thresh.append(nb_vox_above_thresh)
                    sesmap_name.append(session)
                else:
                    print(f"    Warning: {zmap_path} not found")

            except Exception as e:
                print(f"    Error loading {session}: {e}")

        # Get top 4 sessions
        if len(sesmap_vox_above_thresh) >= 4:
            top4_idx = np.argsort(sesmap_vox_above_thresh)[-4:]
            top4_names = np.sort(np.array(sesmap_name)[top4_idx])
        else:
            top4_names = np.array(sesmap_name)
            print(f"    Warning: Only {len(top4_names)} sessions found for {subject}")

        # Load subject-level image
        subj_img_path = op.join(fig_folder, f"{subject}_{condition}.png")
        if op.isfile(subj_img_path):
            subj_images = [np.array(Image.open(subj_img_path))]
        else:
            print(f"    Warning: Subject image not found: {subj_img_path}")
            continue

        # Load top session images
        for sesname in top4_names:
            ses_img_path = op.join(fig_folder, f"{subject}_{sesname}_{condition}.png")
            if op.isfile(ses_img_path):
                subj_images.append(np.array(Image.open(ses_img_path)))
            else:
                print(f"    Warning: Session image not found: {ses_img_path}")

        images.append(subj_images)

    # Create the combined figure
    print(f"  Assembling figure...")
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(4, 9)

    for idx_subj, subject in enumerate(SUBJECTS):
        if idx_subj >= len(images) or len(images[idx_subj]) == 0:
            continue

        # Position for subject-level image (2x2 grid)
        if idx_subj == 0:
            ax1 = fig.add_subplot(gs[0:2, 0:2])
        elif idx_subj == 1:
            ax1 = fig.add_subplot(gs[0:2, 4:6])
        elif idx_subj == 2:
            ax1 = fig.add_subplot(gs[2:4, 0:2])
        elif idx_subj == 3:
            ax1 = fig.add_subplot(gs[2:4, 4:6])

        # Display subject-level image
        ax1.imshow(images[idx_subj][0])
        ax1.axis('off')

        # Display session-level images (4 smaller panels)
        for i in range(min(4, len(images[idx_subj]) - 1)):
            if i == 0:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+0, ((idx_subj%2)*4)+2])
            elif i == 1:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+0, ((idx_subj%2)*4)+3])
            elif i == 2:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+1, ((idx_subj%2)*4)+2])
            elif i == 3:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+1, ((idx_subj%2)*4)+3])

            ax.imshow(images[idx_subj][i+1])
            ax.axis('off')

    # Add title
    fig.suptitle(f"{condition}", fontsize=18, fontweight='bold', x=0.4444)

    # Add colorbar
    _, _, cmap = create_colormap()
    inner_gs = gridspec.GridSpecFromSubplotSpec(4, 8, subplot_spec=gs[:, 8])
    cbar_ax = fig.add_subplot(inner_gs[1:3, 0])
    plt.colorbar(cmap, cax=cbar_ax)

    # Save figure
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_pdf_with_images(image_folder, pdf_filename):
    """Create a PDF with all panel images in a folder.

    Args:
        image_folder (str): Path to folder containing PNG images
        pdf_filename (str): Path to output PDF file
    """
    print(f"\nCreating PDF: {pdf_filename}")

    images = sorted([
        op.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith('.png')
    ])

    if not images:
        print("  Warning: No images found in folder")
        return

    c = canvas.Canvas(pdf_filename)
    size = Image.open(images[0]).size
    c.setPageSize(size)

    for image_path in images:
        print(f"  Adding: {op.basename(image_path)}")
        c.drawImage(image_path, 0, 0, width=size[0], height=size[1])
        c.showPage()

    c.save()
    print(f"  PDF saved: {pdf_filename}")


def main():
    """Main function to generate all annotation panels."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--condition',
        type=str,
        default=None,
        help='Single condition to process (e.g., HIT)'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        default=None,
        help='Comma-separated list of conditions (e.g., HIT,JUMP,Kill)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=op.join(".", "reports", "figures", "full_zmap_plot", "annotations"),
        help='Output directory for panels and PDF'
    )
    parser.add_argument(
        '--skip-individual',
        action='store_true',
        help='Skip generating individual brain maps (only create panels from existing images)'
    )
    parser.add_argument(
        '--skip-panels',
        action='store_true',
        help='Skip generating annotation panels (only create individual images)'
    )
    parser.add_argument(
        '--skip-pdf',
        action='store_true',
        help='Skip generating PDF'
    )

    args = parser.parse_args()

    # Determine which conditions to process
    if args.condition:
        conditions = [args.condition]
    elif args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',')]
    else:
        conditions = DEFAULT_CONDITIONS

    print(f"Processing {len(conditions)} condition(s): {', '.join(conditions)}")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each condition
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Processing condition: {condition}")
        print(f"{'='*60}")

        # Generate individual images for each subject
        if not args.skip_individual:
            for subject in SUBJECTS:
                fig_folder = op.join(
                    ".", "reports", "figures", "full_zmap_plot",
                    subject, condition
                )
                os.makedirs(fig_folder, exist_ok=True)
                create_all_images(subject, condition, fig_folder)

        # Create combined annotation panel
        if not args.skip_panels:
            save_path = op.join(args.output_dir, f"annotations_plot_{condition}.png")
            make_annotation_plot(condition, save_path)

    # Create PDF with all panels
    if not args.skip_pdf:
        pdf_path = op.join(args.output_dir, 'inflated_zmaps_by_annot.pdf')
        create_pdf_with_images(args.output_dir, pdf_path)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
