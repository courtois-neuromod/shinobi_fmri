#!/usr/bin/env python
"""
Generate surface plots comparing two conditions with three-color overlay.

This script creates surface plots showing overlapping significance between two conditions.
Each condition can come from either the shinobi analysis or HCP analysis.

The three-color overlay shows:
- Blue: Significant only for condition 1
- Red: Significant only for condition 2
- Purple: Significant for both conditions

For each subject, creates a surface plot showing the overlay of both conditions.

Usage:
    # Compare shinobi Kill vs HCP Reward
    python viz_condition_comparison.py --cond1 shinobi:Kill --cond2 hcp:reward

    # Compare two shinobi conditions
    python viz_condition_comparison.py --cond1 shinobi:Kill --cond2 shinobi:HealthLoss

    # Compare two HCP conditions
    python viz_condition_comparison.py --cond1 hcp:reward --cond2 hcp:punishment

    # Specify HCP task explicitly
    python viz_condition_comparison.py --cond1 shinobi:Kill --cond2 hcp:reward --hcp-task gambling
"""

import os
import os.path as op
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from nilearn import plotting, image
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from PIL import Image
from tqdm import tqdm
from shinobi_fmri.utils.logger import ShinobiLogger
import logging

try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS
except ImportError:
    print("Warning: config not found. Using defaults.")
    DATA_PATH = "/home/hyruuk/scratch/data"
    SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']


# Default significance threshold
DEFAULT_THRESHOLD = 3.0

# HCP task mapping (for finding the right FFX folder)
HCP_TASKS = {
    'reward': 'gambling',
    'punishment': 'gambling',
    'reward-punishment': 'gambling',
    'punishment-reward': 'gambling',
    'faces': 'social',
    'shapes': 'social',
    'faces-shapes': 'social',
    'match': 'relational',
    'relation': 'relational',
    'relation-match': 'relational',
}


def parse_condition_spec(cond_spec):
    """Parse condition specification in format 'source:condition'.

    Args:
        cond_spec (str): Condition specification like 'shinobi:Kill' or 'hcp:reward'

    Returns:
        tuple: (source, condition) where source is 'shinobi' or 'hcp'
    """
    parts = cond_spec.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid condition format '{cond_spec}'. Use 'source:condition' (e.g., 'shinobi:Kill' or 'hcp:reward')")

    source, condition = parts
    source = source.lower()

    if source not in ['shinobi', 'hcp']:
        raise ValueError(f"Invalid source '{source}'. Must be 'shinobi' or 'hcp'")

    return source, condition


def get_shinobi_zmap_path(subject, condition, data_path):
    """Get path to shinobi subject-level z-map.

    Args:
        subject (str): Subject ID (e.g., 'sub-01')
        condition (str): Condition name (e.g., 'Kill')
        data_path (str): Path to data directory

    Returns:
        str: Path to z-map file
    """
    return op.join(
        data_path,
        "processed",
        "subject-level",
        subject,
        "z_maps",
        f"{subject}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
    )


def find_hcp_task(condition, hcp_task=None):
    """Find which HCP task a condition belongs to.

    Args:
        condition (str): HCP condition name
        hcp_task (str): Optional explicit HCP task name

    Returns:
        str: HCP task name (e.g., 'gambling', 'social', 'relational')
    """
    if hcp_task:
        return hcp_task

    condition_lower = condition.lower()
    if condition_lower in HCP_TASKS:
        return HCP_TASKS[condition_lower]

    # Try to infer from available tasks
    raise ValueError(
        f"Could not determine HCP task for condition '{condition}'. "
        f"Please specify --hcp-task explicitly (gambling, social, or relational)"
    )


def get_hcp_zmap_path(subject, condition, hcp_task, data_path):
    """Get path to HCP FFX z-map.

    Args:
        subject (str): Subject ID (e.g., 'sub-01')
        condition (str): Condition name (e.g., 'reward')
        hcp_task (str): HCP task name (e.g., 'gambling')
        data_path (str): Path to data directory

    Returns:
        str: Path to z-map file
    """
    return op.join(
        data_path,
        "hcp_results",
        subject,
        f"res_stats_{hcp_task}_ffx",
        "stat_maps",
        f"{condition}.nii.gz"
    )


def tstat_to_zstat(t_values, df):
    """Convert t-statistics to z-statistics.

    Args:
        t_values (ndarray): Array of t-statistics
        df (int): Degrees of freedom

    Returns:
        ndarray: Array of z-statistics
    """
    from scipy import stats

    # Convert t to p-value, then p to z
    # For large df, t-distribution approaches normal distribution
    p_values = stats.t.sf(np.abs(t_values), df) * 2  # Two-tailed
    z_values = stats.norm.isf(p_values / 2)  # Two-tailed

    # Preserve sign
    z_values = np.sign(t_values) * z_values

    return z_values


def load_zmap(source, subject, condition, hcp_task, data_path, logger=None):
    """Load z-map for a given condition.

    Args:
        source (str): 'shinobi' or 'hcp'
        subject (str): Subject ID
        condition (str): Condition name
        hcp_task (str): HCP task name (only used if source='hcp')
        data_path (str): Path to data directory
        logger (ShinobiLogger): Logger instance

    Returns:
        Nifti1Image or None: Loaded z-map image, or None if not found
    """
    if source == 'shinobi':
        path = get_shinobi_zmap_path(subject, condition, data_path)
        is_tstat = False
    else:  # hcp
        path = get_hcp_zmap_path(subject, condition, hcp_task, data_path)
        is_tstat = True  # HCP FFX stat_maps contain t-statistics

    if not op.isfile(path):
        if logger:
            logger.warning(f"Map not found: {path}")
        else:
            print(f"Warning: Map not found: {path}")
        return None

    try:
        img = nib.load(path)

        # Convert t-stats to z-stats for HCP
        if is_tstat:
            data = img.get_fdata()

            # Estimate df from HCP data (approximately 40-60 runs per subject)
            # Use conservative estimate of df=50 for FFX
            df = 50

            if logger:
                logger.debug(f"Converting HCP t-statistics to z-statistics (df={df})")

            z_data = tstat_to_zstat(data, df)
            img = nib.Nifti1Image(z_data, img.affine, img.header)

        if logger:
            logger.debug(f"Loaded map: {path}")
        return img
    except Exception as e:
        if logger:
            logger.error(f"Error loading {path}: {e}")
        else:
            print(f"Error loading {path}: {e}")
        return None


def create_overlay_map(img1, img2, threshold=DEFAULT_THRESHOLD, logger=None):
    """Create three-color overlay map from two z-maps with gradient intensities.

    Creates an overlay where:
    - Positive values (1-6): Only condition 1 is significant (gradient by z-score)
    - Negative values (-1 to -6): Only condition 2 is significant (gradient by z-score)
    - Values (7-12): Both conditions are significant (gradient by mean z-score)

    The gradients represent the strength of activation:
    - Lighter colors = lower z-scores (closer to threshold)
    - Darker colors = higher z-scores (stronger activation)

    Args:
        img1 (Nifti1Image): Z-map for condition 1
        img2 (Nifti1Image): Z-map for condition 2
        threshold (float): Significance threshold (default: 3.0)
        logger (ShinobiLogger): Logger instance

    Returns:
        Nifti1Image: Overlay map with coded values
    """
    # Get data arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Check voxel sizes to determine which has lower resolution
    voxsize1 = img1.header.get_zooms()[:3]
    voxsize2 = img2.header.get_zooms()[:3]

    # Calculate voxel volume (larger = lower resolution)
    voxvol1 = np.prod(voxsize1)
    voxvol2 = np.prod(voxsize2)

    # Always resample to the lower resolution (larger voxels) for consistency
    if not np.allclose(voxsize1, voxsize2):
        if voxvol1 > voxvol2:
            # img1 has lower resolution (larger voxels), resample img2 to match
            if logger:
                logger.info(f"Resampling img2 to img1 space ({voxsize2[0]:.1f}mm -> {voxsize1[0]:.1f}mm voxels)")
            img2_resampled = image.resample_to_img(img2, img1, interpolation='continuous')
            data2 = img2_resampled.get_fdata()
            reference_img = img1
        else:
            # img2 has lower resolution, resample img1 to match
            if logger:
                logger.info(f"Resampling img1 to img2 space ({voxsize1[0]:.1f}mm -> {voxsize2[0]:.1f}mm voxels)")
            img1_resampled = image.resample_to_img(img1, img2, interpolation='continuous')
            data1 = img1_resampled.get_fdata()
            reference_img = img2
    else:
        reference_img = img1

    # Create significance masks
    sig1 = np.abs(data1) > threshold
    sig2 = np.abs(data2) > threshold

    # Create overlay map
    overlay = np.zeros_like(data1)

    # Only condition 1 significant: positive values 1-6 scaled by z-score
    only_cond1 = sig1 & ~sig2
    if np.any(only_cond1):
        # Map |z| from [threshold, 6+] to [1, 6]
        z_values = np.abs(data1[only_cond1])
        scaled = 1 + np.clip((z_values - threshold) / 3.0, 0, 1) * 5  # Maps threshold→1, threshold+3→6
        overlay[only_cond1] = scaled

    # Only condition 2 significant: negative values -1 to -6 scaled by z-score
    only_cond2 = sig2 & ~sig1
    if np.any(only_cond2):
        # Map |z| from [threshold, 6+] to [-1, -6]
        z_values = np.abs(data2[only_cond2])
        scaled = -(1 + np.clip((z_values - threshold) / 3.0, 0, 1) * 5)  # Maps threshold→-1, threshold+3→-6
        overlay[only_cond2] = scaled

    # Both significant: positive values 7-12 scaled by mean z-score
    both = sig1 & sig2
    if np.any(both):
        # Use mean of absolute z-scores
        mean_z = (np.abs(data1[both]) + np.abs(data2[both])) / 2.0
        scaled = 7 + np.clip((mean_z - threshold) / 3.0, 0, 1) * 5  # Maps threshold→7, threshold+3→12
        overlay[both] = scaled

    if logger:
        n_only1 = np.sum(only_cond1)
        n_only2 = np.sum(only_cond2)
        n_both = np.sum(both)
        logger.info(f"Overlay: {n_only1} voxels only cond1, {n_only2} voxels only cond2, {n_both} voxels both")

    # Create new image with overlay data using reference image geometry
    overlay_img = nib.Nifti1Image(overlay, reference_img.affine, reference_img.header)

    return overlay_img


def create_three_color_colormap():
    """Create a custom colormap with Blue/Red/Purple for overlays.

    Value ranges (with vmin=-6, vmax=12):
    - -6 to -1: Red gradient for condition 2 (darker = stronger)
    - 0: Not used (no activation)
    - 1 to 6: Blue gradient for condition 1 (darker = stronger)
    - 7 to 12: Purple gradient for both (darker = stronger)

    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap
    """
    # Define color positions for vmin=-6, vmax=12 (range of 18)
    # Position = (value - vmin) / (vmax - vmin) = (value + 6) / 18

    colors = [
        # Red gradient for condition 2 (-6 to -1)
        (0.000, '#8B0000'),  # -6: Dark red (strong activation)
        (0.139, '#FF0000'),  # -3.5: Medium red
        (0.278, '#FF6666'),  # -1: Light red (weak activation)

        # Gap around zero (should not appear due to threshold)
        (0.333, '#FFFFFF'),  # 0: White/transparent

        # Blue gradient for condition 1 (1 to 6)
        (0.389, '#6666FF'),  # 1: Light blue (weak activation)
        (0.528, '#0000FF'),  # 3.5: Medium blue
        (0.667, '#00008B'),  # 6: Dark blue (strong activation)

        # Purple gradient for both conditions (7 to 12)
        (0.722, '#CC66FF'),  # 7: Light purple (weak)
        (0.861, '#9933FF'),  # 9.5: Medium purple
        (1.000, '#6600CC'),  # 12: Dark purple (strong)
    ]

    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'three_color_gradient',
        colors,
        N=256
    )

    return cmap


def plot_single_surface_view(overlay_img, hemisphere, view, cmap, threshold, dpi=150):
    """Plot a single surface view and return as array.

    Args:
        overlay_img: Nifti1Image with overlay data
        hemisphere: 'left' or 'right'
        view: 'lateral' or 'medial'
        cmap: Colormap to use
        threshold: Threshold for display
        dpi: DPI for rendering

    Returns:
        numpy array of the rendered image
    """
    plt.rcParams['figure.dpi'] = dpi

    # Plot single view
    plotting.plot_img_on_surf(
        overlay_img,
        surf_mesh='fsaverage5',
        views=[view],
        hemispheres=[hemisphere],
        inflate=True,
        colorbar=False,
        threshold=threshold,
        vmin=-6,
        vmax=12,
        symmetric_cbar=False,
        cmap=cmap,
        darkness=0.7
    )

    # Get the figure and convert to array
    fig = plt.gcf()
    fig.canvas.draw()

    # Convert to numpy array
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img_array


def apply_surface_mask(overlay_img, img1, img2, z_threshold, logger=None):
    """Apply surface-based masking to overlay to match annotation panel behavior.

    Creates a volume mask based on which voxels would be shown in annotation panels
    when projected to surface, then applies this mask to the overlay.

    Args:
        overlay_img (Nifti1Image): Encoded overlay image
        img1 (Nifti1Image): Raw z-map for condition 1
        img2 (Nifti1Image): Raw z-map for condition 2
        z_threshold (float): Z-score threshold (e.g., 3.0)
        logger (ShinobiLogger): Logger instance

    Returns:
        Nifti1Image: Masked overlay image
    """
    if logger:
        logger.debug("Applying volume-based masking for consistency with annotation panels...")

    # Create volume masks based on threshold (like annotation panels do)
    # Annotation panels use symmetric_cbar=False with threshold, which shows z > threshold
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Create masks: voxels that would be shown in annotation panels
    mask1 = data1 > z_threshold
    mask2 = data2 > z_threshold

    # Combined mask: show overlay where either condition passes threshold
    combined_mask = mask1 | mask2

    # Apply mask to overlay
    overlay_data = overlay_img.get_fdata()
    overlay_data_masked = np.where(combined_mask, overlay_data, 0)

    # Create new image
    masked_overlay_img = nib.Nifti1Image(overlay_data_masked, overlay_img.affine, overlay_img.header)

    if logger:
        voxels_before = np.sum(overlay_data > 0)
        voxels_after = np.sum(overlay_data_masked > 0)
        logger.debug(f"Volume masking applied: {voxels_before} → {voxels_after} voxels")

    return masked_overlay_img


def create_subject_label(subject_id, width, height, dpi=300):
    """Create a subject ID label using matplotlib for proper font scaling.

    Args:
        subject_id (str): Subject ID (e.g., 'sub-01')
        width (int): Width of the label area in pixels
        height (int): Height of the label area in pixels
        dpi (int): DPI for rendering

    Returns:
        PIL.Image: Label as PIL Image
    """
    from PIL import Image
    import io

    # Convert pixels to inches
    fig_width = width / dpi
    fig_height = height / dpi

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Add text centered
    ax.text(0.5, 0.5, subject_id, ha='center', va='center',
            fontsize=24, fontweight='normal',
            transform=ax.transAxes)

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    label_img = Image.open(buf)
    label_img = label_img.copy()
    buf.close()
    plt.close(fig)

    return label_img


def create_four_subject_panel(subject_images, save_path, legend_data, dpi=300, logger=None):
    """Create a 2x2 panel with all 4 subjects and legend on the right.

    Args:
        subject_images (dict): Dict mapping subject IDs to image arrays
        save_path (str): Path to save panel
        legend_data (Image): PIL Image of the legend
        dpi (int): Output DPI
        logger: Logger instance
    """
    from PIL import Image

    if logger:
        logger.debug("Creating 4-subject panel...")

    # Expected subject order for 2x2 grid
    subjects_order = ['sub-01', 'sub-02', 'sub-04', 'sub-06']

    # Load images
    imgs = []
    for subj in subjects_order:
        if subj in subject_images:
            imgs.append(Image.fromarray(subject_images[subj]))
        else:
            # Create blank if missing
            imgs.append(Image.new('RGB', (600, 300), 'white'))

    # Get dimensions
    img_width = imgs[0].width
    img_height = imgs[0].height

    # Legend dimensions
    legend_width = legend_data.width
    legend_height = legend_data.height

    # Calculate panel dimensions
    inter_subject_spacing = 80  # Space between different subjects (sub-01 vs sub-02)
    intra_subject_spacing = 5   # Space within each subject's 4 views
    edge_spacing = 20
    label_height = 80  # Space for subject ID labels

    # 2x2 grid of subjects
    grid_width = (img_width * 2) + inter_subject_spacing
    grid_height = (img_height * 2) + inter_subject_spacing + (2 * label_height)  # Add space for 2 labels

    # Total dimensions
    total_width = grid_width + edge_spacing * 2 + legend_width + edge_spacing
    total_height = grid_height + edge_spacing * 2

    # Create panel
    panel = Image.new('RGB', (total_width, total_height), 'white')

    # Place images in 2x2 grid with centered labels above each subject
    # Layout:
    # [sub-01 label centered]    [sub-02 label centered]
    # [sub-01 images]            [sub-02 images]
    # [sub-04 label centered]    [sub-06 label centered]
    # [sub-04 images]            [sub-06 images]

    positions = [
        (edge_spacing, edge_spacing + label_height),  # sub-01: top-left
        (edge_spacing + img_width + inter_subject_spacing, edge_spacing + label_height),  # sub-02: top-right
        (edge_spacing, edge_spacing + img_height + inter_subject_spacing + 2 * label_height),  # sub-04: bottom-left
        (edge_spacing + img_width + inter_subject_spacing, edge_spacing + img_height + inter_subject_spacing + 2 * label_height)  # sub-06: bottom-right
    ]

    for idx, (img, pos) in enumerate(zip(imgs, positions)):
        # Paste image
        panel.paste(img, pos)

        # Create and paste subject label using matplotlib (for proper font scaling at high DPI)
        subj_label = subjects_order[idx]
        label_img = create_subject_label(subj_label, img_width, label_height, dpi)

        # Center the label above the images
        label_x = pos[0]
        label_y = pos[1] - label_height

        # Resize label if needed to fit exactly in the label area
        if label_img.width != img_width or label_img.height != label_height:
            # Paste at center of label area
            label_offset_x = pos[0] + (img_width - label_img.width) // 2
            label_offset_y = pos[1] - label_height + (label_height - label_img.height) // 2
            panel.paste(label_img, (label_offset_x, label_offset_y), label_img if label_img.mode == 'RGBA' else None)
        else:
            panel.paste(label_img, (label_x, label_y), label_img if label_img.mode == 'RGBA' else None)

    # Place legend on the right, centered vertically
    legend_y = (total_height - legend_height) // 2
    legend_x = grid_width + edge_spacing * 2
    panel.paste(legend_data, (legend_x, legend_y))

    # Save
    panel.save(save_path, dpi=(dpi, dpi))

    if logger:
        logger.info(f"Saved panel: {save_path}")


def plot_overlay_surface(overlay_img, save_path, img1=None, img2=None, z_threshold=3.0, threshold=2.5, dpi=300, logger=None):
    """Plot overlay map on inflated brain surface.

    Args:
        overlay_img (Nifti1Image): Overlay map with three-color coding
        save_path (str): Path to save the image
        img1 (Nifti1Image): Raw z-map for condition 1 (for masking)
        img2 (Nifti1Image): Raw z-map for condition 2 (for masking)
        z_threshold (float): Z-score threshold for raw maps (default: 3.0)
        threshold (float): Threshold for display (default: 1.0)
        dpi (int): DPI of output image
        logger (ShinobiLogger): Logger instance
    """
    if logger:
        logger.debug("Rendering individual surface views...")

    # Apply surface-based masking if raw z-maps are provided
    if img1 is not None and img2 is not None:
        overlay_img = apply_surface_mask(overlay_img, img1, img2, z_threshold, logger)

    # Create custom three-color colormap (Blue/Red/Purple)
    cmap = create_three_color_colormap()

    # Render each view separately at high DPI for quality (matching annotation panels)
    render_dpi = 300

    # Render surfaces in order: left lateral, right lateral, left medial, right medial
    left_lateral = plot_single_surface_view(overlay_img, 'left', 'lateral', cmap, threshold, render_dpi)
    right_lateral = plot_single_surface_view(overlay_img, 'right', 'lateral', cmap, threshold, render_dpi)
    left_medial = plot_single_surface_view(overlay_img, 'left', 'medial', cmap, threshold, render_dpi)
    right_medial = plot_single_surface_view(overlay_img, 'right', 'medial', cmap, threshold, render_dpi)

    # Convert arrays to PIL Images and crop whitespace
    from PIL import Image

    def crop_whitespace(img_array):
        """Crop whitespace from image array."""
        img = Image.fromarray(img_array)
        # Convert to grayscale to find non-white pixels
        gray = img.convert('L')
        # Find bounding box of non-white content
        bbox = gray.point(lambda x: 0 if x > 250 else 255).getbbox()
        if bbox:
            return img.crop(bbox)
        return img

    ll_img = crop_whitespace(left_lateral)
    rl_img = crop_whitespace(right_lateral)
    lm_img = crop_whitespace(left_medial)
    rm_img = crop_whitespace(right_medial)

    # Create compact 2x2 grid layout
    # Layout: [Left Lateral | Right Lateral]
    #         [Left Medial | Right Medial]
    spacing = 5  # Minimal spacing in pixels

    # Make all brain images the same height
    max_height = max(ll_img.height, rl_img.height, lm_img.height, rm_img.height)

    def resize_to_height(img, target_height):
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, target_height), Image.LANCZOS)

    ll_img = resize_to_height(ll_img, max_height)
    rl_img = resize_to_height(rl_img, max_height)
    lm_img = resize_to_height(lm_img, max_height)
    rm_img = resize_to_height(rm_img, max_height)

    # Calculate dimensions for 2x2 grid
    # Two rows of brain images
    row_width = max(ll_img.width + rl_img.width, lm_img.width + rm_img.width) + spacing
    grid_height = 2 * max_height + spacing

    # Calculate total dimensions
    total_width = row_width + spacing
    total_height = grid_height + spacing

    # Create final composite image
    final_img = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images in 2x2 grid with minimal spacing
    y_offset = spacing

    # Top row: Left Lateral | Right Lateral
    x_offset = spacing
    final_img.paste(ll_img, (x_offset, y_offset))
    x_offset += ll_img.width + spacing
    final_img.paste(rl_img, (x_offset, y_offset))

    # Bottom row: Left Medial | Right Medial
    y_offset += max_height + spacing
    x_offset = spacing
    final_img.paste(lm_img, (x_offset, y_offset))
    x_offset += lm_img.width + spacing
    final_img.paste(rm_img, (x_offset, y_offset))

    # Save final image
    if save_path:
        final_img.save(save_path, dpi=(dpi, dpi))
        if logger:
            logger.debug(f"Saved compact overlay surface plot: {save_path}")

    # Return as numpy array for panel creation
    return np.array(final_img)


def get_condition_color(source, condition):
    """Get the color for a condition based on beta correlations plot color scheme.

    Args:
        source (str): 'shinobi' or 'hcp'
        condition (str): Condition name

    Returns:
        str: Hex color code
    """
    import seaborn as sns

    # Use same color scheme as beta_correlations_plot.py
    palette = sns.color_palette("Set2")
    palette_dark = sns.color_palette("Dark2")

    if source == 'shinobi':
        # Shinobi events use Set2 palette[1]
        return mcolors.rgb2hex(palette[1])
    else:
        # HCP tasks use Dark2 palette
        # Gambling (reward/punishment): palette_dark[2]
        # Social: palette_dark[5]
        # Relational: palette_dark[6]
        hcp_colors = {
            'gambling': palette_dark[2],
            'social': palette_dark[5],
            'relational': palette_dark[6]
        }
        # Find task for this condition
        condition_lower = condition.lower()
        if condition_lower in ['reward', 'punishment']:
            task = 'gambling'
        elif condition_lower in ['faces', 'shapes']:
            task = 'social'
        elif condition_lower in ['match', 'relation']:
            task = 'relational'
        else:
            task = 'gambling'  # default

        return mcolors.rgb2hex(hcp_colors[task])


def create_legend_image(source1, cond1, source2, cond2, dpi=150):
    """Create a vertical legend with Blue/Red/Purple gradients and task-colored labels.

    Args:
        source1 (str): Source for condition 1 ('shinobi' or 'hcp')
        cond1 (str): Condition 1 name
        source2 (str): Source for condition 2 ('shinobi' or 'hcp')
        cond2 (str): Condition 2 name
        dpi (int): DPI of output image

    Returns:
        PIL.Image: Legend as PIL Image
    """
    from PIL import Image
    import io

    # Get colors for condition labels (task-specific)
    color1 = get_condition_color(source1, cond1)
    color2 = get_condition_color(source2, cond2)

    # Vertical layout - much larger to accommodate bigger text
    fig, ax = plt.subplots(figsize=(8, 18), dpi=dpi)
    ax.axis('off')

    # Create gradient colorbars - inverted so dark is at top, light at bottom
    # Gradient values: 0 (dark) at top → 1 (light) at bottom
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    # Bar width (half of previous)
    bar_width = 0.075
    bar_x = 0.05

    # Blue gradient for condition 1
    ax_cond1 = fig.add_axes([bar_x, 0.68, bar_width, 0.25])
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cond1', ['#00008B', '#6666FF'])
    ax_cond1.imshow(gradient, aspect='auto', cmap=cmap1, extent=[0, 1, 0, 1])
    ax_cond1.set_xticks([])
    # Add ticks for z-values (bottom=3 light, middle=4.5, top=6 dark)
    ax_cond1.set_yticks([0, 0.5, 1])
    ax_cond1.set_yticklabels(['3', '4.5', '6'], fontsize=28)
    ax_cond1.tick_params(axis='y', length=8, width=2, pad=5)
    for spine in ax_cond1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
    # Add label directly on this axes, centered vertically at 0.5
    ax_cond1.text(1.3, 0.5, cond1, ha='left', va='center', fontsize=45,
                  fontweight='bold', color=color1, transform=ax_cond1.transAxes)

    # Red gradient for condition 2
    ax_cond2 = fig.add_axes([bar_x, 0.39, bar_width, 0.25])
    cmap2 = mcolors.LinearSegmentedColormap.from_list('cond2', ['#8B0000', '#FF6666'])
    ax_cond2.imshow(gradient, aspect='auto', cmap=cmap2, extent=[0, 1, 0, 1])
    ax_cond2.set_xticks([])
    ax_cond2.set_yticks([0, 0.5, 1])
    ax_cond2.set_yticklabels(['3', '4.5', '6'], fontsize=28)
    ax_cond2.tick_params(axis='y', length=8, width=2, pad=5)
    for spine in ax_cond2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
    # Add label directly on this axes, centered vertically at 0.5
    ax_cond2.text(1.3, 0.5, cond2, ha='left', va='center', fontsize=45,
                  fontweight='bold', color=color2, transform=ax_cond2.transAxes)

    # Purple gradient for both conditions
    ax_both = fig.add_axes([bar_x, 0.10, bar_width, 0.25])
    cmap_both = mcolors.LinearSegmentedColormap.from_list('both', ['#6600CC', '#CC66FF'])
    ax_both.imshow(gradient, aspect='auto', cmap=cmap_both, extent=[0, 1, 0, 1])
    ax_both.set_xticks([])
    ax_both.set_yticks([0, 0.5, 1])
    ax_both.set_yticklabels(['3', '4.5', '6'], fontsize=28)
    ax_both.tick_params(axis='y', length=8, width=2, pad=5)
    for spine in ax_both.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
    # Add label directly on this axes, centered vertically at 0.5
    ax_both.text(1.3, 0.5, 'Both', ha='left', va='center', fontsize=45,
                 fontweight='bold', color='black', transform=ax_both.transAxes)

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    legend_img = Image.open(buf)
    legend_img = legend_img.copy()  # Make a copy before closing buffer
    buf.close()
    plt.close(fig)

    return legend_img


def main():
    """Main function to generate condition comparison plots."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--cond1',
        type=str,
        default=None,
        help='First condition in format "source:condition" (e.g., "shinobi:Kill" or "hcp:reward")'
    )
    parser.add_argument(
        '--cond2',
        type=str,
        default=None,
        help='Second condition in format "source:condition" (e.g., "shinobi:HealthLoss" or "hcp:punishment")'
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all predefined comparisons (Kill vs reward, HealthLoss vs punishment, etc.)'
    )
    parser.add_argument(
        '--hcp-task',
        type=str,
        default=None,
        help='HCP task name if using HCP conditions (gambling, social, relational). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'Significance threshold for z-maps (default: {DEFAULT_THRESHOLD})'
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
        help='Output directory for plots (default: ./reports/figures/condition_comparison/<cond1>_vs_<cond2>/)'
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

    # Validate arguments
    if args.run_all and (args.cond1 or args.cond2):
        parser.error("Cannot specify both --run-all and individual conditions")
    if not args.run_all and (not args.cond1 or not args.cond2):
        parser.error("Must specify either --run-all or both --cond1 and --cond2")

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="CondComparison",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        # Define comparison pairs
        if args.run_all:
            comparisons = [
                ('shinobi:Kill', 'hcp:reward'),
                ('shinobi:HealthLoss', 'hcp:punishment'),
                ('shinobi:RIGHT', 'shinobi:JUMP'),
                ('shinobi:LEFT', 'shinobi:HIT'),
            ]
        else:
            comparisons = [(args.cond1, args.cond2)]

        logger.info(f"Processing {len(comparisons)} comparison(s)")
        logger.info(f"Subjects: {', '.join(SUBJECTS)}")
        logger.info(f"Threshold: |z| > {args.threshold}\n")

        # Process each comparison
        for cond_spec1, cond_spec2 in comparisons:
            # Parse condition specifications
            source1, cond1 = parse_condition_spec(cond_spec1)
            source2, cond2 = parse_condition_spec(cond_spec2)

            logger.info(f"\n{'='*70}")
            logger.info(f"Comparing {source1}:{cond1} vs {source2}:{cond2}")
            logger.info(f"{'='*70}\n")

            # Determine HCP task if needed
            hcp_task1 = None
            hcp_task2 = None
            if source1 == 'hcp':
                hcp_task1 = find_hcp_task(cond1, args.hcp_task)
                logger.info(f"Using HCP task '{hcp_task1}' for condition 1")
            if source2 == 'hcp':
                hcp_task2 = find_hcp_task(cond2, args.hcp_task)
                logger.info(f"Using HCP task '{hcp_task2}' for condition 2")

            # Use single output directory for all panels
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = op.join(".", "reports", "figures", "condition_comparison")
            os.makedirs(output_dir, exist_ok=True)

            if len(comparisons) == 1:
                logger.info(f"Output directory: {output_dir}\n")

            # Get task colors for conditions
            color1 = get_condition_color(source1, cond1)
            color2 = get_condition_color(source2, cond2)
            logger.info(f"Using colors: {cond1}={color1}, {cond2}={color2}")

            # Create legend as PIL Image
            legend_img = create_legend_image(source1, cond1, source2, cond2)
            logger.debug("Created legend image")

            # Collect all subjects
            subject_images = {}
            for subject in tqdm(SUBJECTS, desc=f"Processing {cond1} vs {cond2}"):
                logger.debug(f"Processing {subject}")

                # Load z-maps
                img1 = load_zmap(source1, subject, cond1, hcp_task1, args.data_path, logger)
                img2 = load_zmap(source2, subject, cond2, hcp_task2, args.data_path, logger)

                if img1 is None or img2 is None:
                    logger.warning(f"Skipping {subject} due to missing data")
                    continue

                # Create overlay
                overlay_img = create_overlay_map(img1, img2, args.threshold, logger)

                # Plot overlay and get image array (don't save individual files)
                img_array = plot_overlay_surface(overlay_img, None,
                                   img1=img1, img2=img2, z_threshold=args.threshold,
                                   threshold=2.5, logger=logger)

                subject_images[subject] = img_array

            # Create 4-subject panel
            panel_path = op.join(output_dir, f"{cond1}_vs_{cond2}_panel.png")
            create_four_subject_panel(subject_images, panel_path, legend_img,
                                    logger=logger)

        # Print summary
        logger.info("\nAll done!")
        logger.info(f"Created {len(comparisons)} comparison panel(s)")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
