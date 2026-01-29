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
import warnings
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
from tqdm import tqdm
from shinobi_fmri.utils.logger import AnalysisLogger
import logging

# Filter specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy", message="Warning: 'partition' will ignore the 'mask' of the MaskedArray")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="The `darkness` parameter will be deprecated")

try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS
except ImportError:
    print("Warning: config not found. Using defaults.")
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


def crop_whitespace(img):
    """Crop whitespace from PIL Image."""
    # Convert to grayscale to find non-white pixels
    gray = img.convert('L')
    # Find bounding box of non-white content
    bbox = gray.point(lambda x: 0 if x > 250 else 255).getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def plot_inflated_zmap(img, save_path=None, title=None, colorbar=True, vmax=6, threshold=2.3, dpi=300):
    """Plot inflated brain surface map and save the image.

    Creates a composite of 4 views (Lateral/Medial x Left/Right) with minimal spacing.

    Args:
        img (str or Nifti1Image): Path to image or nibabel image object
        save_path (str): Path to save the image
        title (str): Title of the image
        colorbar (bool): Whether to include a colorbar (ignored in this version, always False)
        vmax (float): Maximum value for the colormap
        threshold (float): Threshold value for displaying voxels
        dpi (int): DPI of the output image
    """
    plt.rcParams['figure.dpi'] = dpi

    # Use standard cold_hot colormap without grey zone
    cmap = nilearn_cmaps['cold_hot']

    # 1. Render 4 views independently
    views = [
        ('left', 'lateral'),
        ('right', 'lateral'),
        ('left', 'medial'),
        ('right', 'medial')
    ]
    
    rendered_imgs = []
    
    for hemi, view in views:
        # Plot single view
        plotting.plot_img_on_surf(
            img,
            surf_mesh='fsaverage5',
            views=[view],
            hemispheres=[hemi],
            inflate=True,
            colorbar=False,
            threshold=threshold,
            vmax=vmax,
            vmin=-vmax,
                            symmetric_cbar=False,
                            cmap=cmap,
                            darkness=None
                        )
                    # Capture the figure
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Convert to PIL Image
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = Image.fromarray(data.reshape((h, w, 3)))
        
        plt.close(fig)
        
        # Crop whitespace immediately
        rendered_imgs.append(crop_whitespace(im))

    # 2. Resize all to the same height (based on max height)
    max_height = max(im.height for im in rendered_imgs)
    
    resized_imgs = []
    for im in rendered_imgs:
        if im.height != max_height:
            ratio = max_height / im.height
            new_width = int(im.width * ratio)
            im = im.resize((new_width, max_height), Image.LANCZOS)
        else:
            im = im.copy() # Just to be safe
        resized_imgs.append(im)
    
    ll, rl, lm, rm = resized_imgs
    
    # 3. Stitch in 2x2 grid with minimal spacing
    spacing = 5
    
    # Calculate dimensions
    row1_w = ll.width + spacing + rl.width
    row2_w = lm.width + spacing + rm.width
    total_w = max(row1_w, row2_w)
    total_h = max_height * 2 + spacing
    
    # Create composite image
    final_img = Image.new('RGB', (total_w, total_h), 'white')
    
    # Paste Row 1: Left Lateral | Right Lateral (Centered)
    r1_x = (total_w - row1_w) // 2
    final_img.paste(ll, (r1_x, 0))
    final_img.paste(rl, (r1_x + ll.width + spacing, 0))
    
    # Paste Row 2: Left Medial | Right Medial (Centered)
    r2_x = (total_w - row2_w) // 2
    final_img.paste(lm, (r2_x, max_height + spacing))
    final_img.paste(rm, (r2_x + lm.width + spacing, max_height + spacing))

    # 4. Add Title using Matplotlib (to match previous font styling)
    # Calculate new figure size in inches
    # Add extra space at top for title
    title_height_fraction = 0.25 if title else 0.0
    
    # We want the image part to have the correct aspect ratio
    img_aspect = total_w / total_h
    
    # Base width in inches (arbitrary, affects resolution combined with DPI)
    fig_width = 8 
    fig_height = fig_width / img_aspect
    
    # Adjust for title space
    total_fig_height = fig_height * (1 + title_height_fraction)
    
    fig = plt.figure(figsize=(fig_width, total_fig_height), dpi=dpi)
    
    # Add axes for the image, leaving space at top
    ax_h = 1.0 / (1 + title_height_fraction)
    ax = fig.add_axes([0, 0, 1, ax_h])
    ax.imshow(final_img)
    ax.axis('off')
    
    if title:
        # Determine fontsize
        if 'sub' in title:
            fontsize = 48
        elif 'ses' in title:
            fontsize = 64
        else:
            fontsize = 40
        
        # Add title in the top margin area
        # 1.0 is top of image axes. We want to go higher.
        # Figure coordinates: 0,0 is bottom-left, 1,1 is top-right.
        # Top of image is at y = ax_h.
        title_y = ax_h + (1 - ax_h) * 0.5 # Center in the top margin
        fig.text(0.5, title_y, title, ha='center', va='center', fontsize=fontsize)

    # Save
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    
    plt.close(fig)


def create_all_images(subject, condition, fig_folder, data_path=DATA_PATH, use_corrected_maps=False, force=False, pbar=None, logger=None):
    """Create all individual brain map images for a subject and condition.
    
    Low-level features (luminance, optical_flow, etc.) are always included.
    The source is always the 'processed/' directory.
    """
    # Always use processed directory (low-level features are now default)
    output_dir = "processed"

    # Create subject-level z-map
    # New structure: processed/subject-level/sub-XX/z_maps/
    # By default, use raw maps; use corrected if use_corrected_maps=True
    if use_corrected_maps:
        sublevel_zmap_path = op.join(
            data_path,
            output_dir,
            "subject-level",
            subject,
            "z_maps",
            f"{subject}_task-shinobi_contrast-{condition}_desc-corrected_stat-z.nii.gz"
        )
        # Fall back to raw map if corrected doesn't exist
        if not op.isfile(sublevel_zmap_path):
            sublevel_zmap_path = op.join(
                data_path,
                output_dir,
                "subject-level",
                subject,
                "z_maps",
                f"{subject}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
            )
            if logger and op.isfile(sublevel_zmap_path):
                logger.debug(f"Corrected map not found, using raw map for {subject} {condition}")
    else:
        sublevel_zmap_path = op.join(
            data_path,
            output_dir,
            "subject-level",
            subject,
            "z_maps",
            f"{subject}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
        )

    sublevel_save_path = op.join(fig_folder, f"{subject}_{condition}.png")

    if force or not op.isfile(sublevel_save_path):
        if op.isfile(sublevel_zmap_path):
            if logger:
                logger.debug(f"Plotting subject-level map: {sublevel_zmap_path}")
            plot_inflated_zmap(
                sublevel_zmap_path,
                save_path=sublevel_save_path,
                title=f"{subject}",
                colorbar=False
            )
        else:
            # Create blank image if z-map doesn't exist
            if logger:
                logger.debug(f"Subject-level map missing, creating blank: {sublevel_save_path}")
            create_blank_image(sublevel_save_path, fig_folder)

    if pbar:
        pbar.update(1)

    # Create session-level z-maps
    ses_dir = op.join(data_path, "shinobi", subject)
    if op.exists(ses_dir):
        ses_list = sorted(os.listdir(ses_dir))

        for session in ses_list:
            # New structure: processed/session-level/sub-XX/ses-YY/z_maps/
            # By default, use raw maps; use corrected if use_corrected_maps=True
            if use_corrected_maps:
                zmap_path = op.join(
                    data_path,
                    output_dir,
                    "session-level",
                    subject,
                    session,
                    "z_maps",
                    f"{subject}_{session}_task-shinobi_contrast-{condition}_desc-corrected_stat-z.nii.gz"
                )
                # Fall back to raw map if corrected doesn't exist
                if not op.isfile(zmap_path):
                    zmap_path = op.join(
                        data_path,
                        output_dir,
                        "session-level",
                        subject,
                        session,
                        "z_maps",
                        f"{subject}_{session}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
                    )
                    if logger and op.isfile(zmap_path):
                        logger.debug(f"Corrected map not found, using raw map for {subject} {session} {condition}")
            else:
                zmap_path = op.join(
                    data_path,
                    output_dir,
                    "session-level",
                    subject,
                    session,
                    "z_maps",
                    f"{subject}_{session}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
                )

            save_path = op.join(fig_folder, f"{subject}_{session}_{condition}.png")

            if force or not op.isfile(save_path):
                if op.isfile(zmap_path):
                    if logger:
                        logger.debug(f"Plotting session-level map: {zmap_path}")
                    plot_inflated_zmap(
                        zmap_path,
                        save_path=save_path,
                        title=f"{session}",
                        colorbar=False
                    )
                else:
                    # Create blank image if z-map doesn't exist
                    if logger:
                        logger.debug(f"Session-level map missing, creating blank: {save_path}")
                    create_blank_image(save_path, fig_folder)

            if pbar:
                pbar.update(1)


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


def make_annotation_plot(condition, save_path, data_path=DATA_PATH, use_corrected_maps=False, pbar=None, logger=None):
    """Create a combined panel showing all subjects for one annotation.

    Low-level features are always included.
    Source is always the 'processed/' directory.
    """
    # Always use processed directory (low-level features are now default)
    output_dir = "processed"

    images = []

    for idx_subj, subject in enumerate(SUBJECTS):
        fig_folder = op.join(".", "reports", "figures", "full_zmap_plot", subject, condition)

        # Find top 4 session maps based on number of voxels above threshold
        ses_dir = op.join(data_path, "shinobi", subject)
        if not op.exists(ses_dir):
            if logger:
                logger.warning(f"Session directory not found for {subject}")
            continue

        ses_list = sorted(os.listdir(ses_dir))
        sesmap_vox_above_thresh = []
        sesmap_name = []

        for session in ses_list:
            try:
                # New structure: processed/session-level/sub-XX/ses-YY/z_maps/
                # Use raw maps by default; use corrected if use_corrected_maps=True
                if use_corrected_maps:
                    zmap_path = op.join(
                        data_path,
                        output_dir,
                        "session-level",
                        subject,
                        session,
                        "z_maps",
                        f"{subject}_{session}_task-shinobi_contrast-{condition}_desc-corrected_stat-z.nii.gz"
                    )
                    # Fall back to raw map if corrected doesn't exist
                    if not op.isfile(zmap_path):
                        zmap_path = op.join(
                            data_path,
                            output_dir,
                            "session-level",
                            subject,
                            session,
                            "z_maps",
                            f"{subject}_{session}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
                        )
                else:
                    zmap_path = op.join(
                        data_path,
                        output_dir,
                        "session-level",
                        subject,
                        session,
                        "z_maps",
                        f"{subject}_{session}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
                    )

                if op.isfile(zmap_path):
                    img = nib.load(zmap_path).get_fdata()
                    # Count voxels with |z| > 3
                    nb_vox_above_thresh = np.sum(np.abs(img.flatten()) > 3)
                    sesmap_vox_above_thresh.append(nb_vox_above_thresh)
                    sesmap_name.append(session)
                else:
                    if logger:
                        logger.debug(f"{zmap_path} not found")

            except Exception as e:
                if logger:
                    logger.error(f"Error loading {session}: {e}")

        # Log all available sessions and their voxel counts
        if logger:
            logger.debug(f"{subject} {condition}: All sessions - {list(zip(sesmap_name, sesmap_vox_above_thresh))}")

        # Get top 4 sessions
        if len(sesmap_vox_above_thresh) >= 4:
            top4_idx = np.argsort(sesmap_vox_above_thresh)[-4:]
            top4_names = np.sort(np.array(sesmap_name)[top4_idx])
            top4_voxcounts = np.array(sesmap_vox_above_thresh)[top4_idx]
            if logger:
                logger.info(f"{subject} {condition}: Using top 4 sessions - {list(zip(top4_names, top4_voxcounts))}")
            else:
                print(f"{subject} {condition}: Top 4 sessions - {list(zip(top4_names, top4_voxcounts))}")
        else:
            top4_names = np.array(sesmap_name)
            if logger:
                logger.warning(f"Only {len(top4_names)} sessions found for {subject}")
            else:
                print(f"WARNING: Only {len(top4_names)} sessions found for {subject}")

        # Load subject-level image
        subj_img_path = op.join(fig_folder, f"{subject}_{condition}.png")
        if op.isfile(subj_img_path):
            subj_images = [np.array(Image.open(subj_img_path))]
        else:
            if logger:
                logger.warning(f"Subject image not found: {subj_img_path}")
            continue

        # Load top session images
        for sesname in top4_names:
            ses_img_path = op.join(fig_folder, f"{subject}_{sesname}_{condition}.png")
            if op.isfile(ses_img_path):
                subj_images.append(np.array(Image.open(ses_img_path)))
            else:
                if logger:
                    logger.warning(f"Session image not found: {ses_img_path}")

        images.append(subj_images)

    # Create the combined figure
    if logger:
        logger.info(f"Assembling figure for {condition}")
        
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
    fig.suptitle(f"{condition}", fontsize=32, x=0.4444)

    # Add colorbar
    _, _, cmap = create_colormap()
    inner_gs = gridspec.GridSpecFromSubplotSpec(4, 8, subplot_spec=gs[:, 8])
    cbar_ax = fig.add_subplot(inner_gs[1:3, 0])
    cbar = plt.colorbar(cmap, cax=cbar_ax)

    # Add z label above colorbar
    cbar_ax.text(0.5, 1.05, r'$z$', ha='center', va='bottom', fontsize=12,
                 transform=cbar_ax.transAxes)
    # Save figure
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if pbar:
        pbar.update(1)


def create_pdf_with_images(image_folder, pdf_filename, pbar=None):
    """Create a PDF with all panel images in a folder.

    Args:
        image_folder (str): Path to folder containing PNG images
        pdf_filename (str): Path to output PDF file
        pbar (tqdm): Optional progress bar to update
    """
    images = sorted([
        op.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith('.png')
    ])

    if not images:
        print("Warning: No images found in folder")
        return

    c = canvas.Canvas(pdf_filename)
    size = Image.open(images[0]).size
    c.setPageSize(size)

    for image_path in images:
        c.drawImage(image_path, 0, 0, width=size[0], height=size[1])
        c.showPage()
        if pbar:
            pbar.update(1)

    c.save()

    if pbar:
        pbar.set_description("PDF created")


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
        '--data-path',
        type=str,
        default=DATA_PATH,
        help=f'Path to data directory (default: {DATA_PATH})'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for panels and PDF (default: reports/<fig_dir>/full_zmap_plot/annotations)'
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
    parser.add_argument(
        '--use-raw-maps',
        action='store_true',
        help='Use raw uncorrected z-maps instead of corrected maps (default: use corrected maps)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of images even if they already exist'
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

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = op.join(".", "reports", "figures", "full_zmap_plot", "annotations")

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="Viz_panels",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        # Determine which conditions to process
        if args.condition:
            conditions = [args.condition]
        elif args.conditions:
            conditions = [c.strip() for c in args.conditions.split(',')]
        else:
            from shinobi_fmri.config import CONDITIONS as GAME_CONDITIONS, LOW_LEVEL_CONDITIONS
            conditions = GAME_CONDITIONS + LOW_LEVEL_CONDITIONS
            if args.exclude_low_level:
                conditions = GAME_CONDITIONS

        logger.info(f"Processing {len(conditions)} condition(s): {', '.join(conditions)}")
        logger.info(f"Subjects: {', '.join(SUBJECTS)}")
        logger.info(f"Using {'raw uncorrected' if args.use_raw_maps else 'corrected'} z-maps")
        logger.info(f"Including low-level features: {not args.exclude_low_level}")
        logger.info(f"Output directory: {args.output_dir}\n")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Calculate total work for progress tracking
        total_individual_images = 0
        if not args.skip_individual:
            for condition in conditions:
                for subject in SUBJECTS:
                    # Count: 1 subject-level + N session-level images
                    ses_dir = op.join(args.data_path, "shinobi", subject)
                    if op.exists(ses_dir):
                        n_sessions = len(os.listdir(ses_dir))
                        total_individual_images += 1 + n_sessions  # subject + sessions
                    else:
                        total_individual_images += 1  # just subject

        # Process each condition
        for condition in conditions:
            logger.info(f"Processing condition: {condition}")

            # Generate individual images for each subject
            if not args.skip_individual:
                with tqdm(total=total_individual_images // len(conditions),
                          desc=f"Creating images for {condition}",
                          unit="img") as pbar:
                    for subject in SUBJECTS:
                        fig_folder = op.join(
                            ".", "reports", "figures", "full_zmap_plot",
                            subject, condition
                        )
                        os.makedirs(fig_folder, exist_ok=True)
                        create_all_images(subject, condition, fig_folder,
                                          data_path=args.data_path, use_corrected_maps=not args.use_raw_maps,
                                          force=args.force, pbar=pbar, logger=logger)

            # Create combined annotation panel
            if not args.skip_panels:
                # Add suffix to indicate corrected vs raw maps
                map_type = "raw" if args.use_raw_maps else "corrected"
                save_path = op.join(args.output_dir, f"annotations_plot_{condition}_{map_type}.png")
                with tqdm(total=1, desc=f"Creating panel for {condition}", unit="panel") as pbar:
                    make_annotation_plot(condition, save_path,
                                         data_path=args.data_path, use_corrected_maps=not args.use_raw_maps,
                                         pbar=pbar, logger=logger)

        # Create PDF with all panels
        if not args.skip_pdf:
            # Add suffix to indicate corrected vs raw maps
            map_type = "raw" if args.use_raw_maps else "corrected"
            pdf_path = op.join(args.output_dir, f'inflated_zmaps_by_annot_{map_type}.pdf')
            panel_images = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
            with tqdm(total=len(panel_images), desc="Creating PDF", unit="page") as pbar:
                create_pdf_with_images(args.output_dir, pdf_path, pbar=pbar)

        # Print summary of outputs
        logger.info("All done!")

        if not args.skip_individual:
            logger.info(f"Individual brain maps saved to ./reports/figures/full_zmap_plot/<subject>/<condition>/")

        if not args.skip_panels:
            logger.info(f"Annotation panels saved to {op.abspath(args.output_dir)}/")
            for condition in conditions:
                panel_file = f"annotations_plot_{condition}.png"
                if op.exists(op.join(args.output_dir, panel_file)):
                    logger.debug(f"Saved: {panel_file}")

        if not args.skip_pdf:
            pdf_path = op.join(args.output_dir, 'inflated_zmaps_by_annot.pdf')
            if op.exists(pdf_path):
                logger.info(f"PDF compilation saved to {op.abspath(pdf_path)} ({len(panel_images)} pages)")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
