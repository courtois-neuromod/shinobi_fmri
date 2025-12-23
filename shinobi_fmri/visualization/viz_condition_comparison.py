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


def plot_overlay_surface(overlay_img, save_path, title, legend_img_path, color1, color2, img1=None, img2=None, z_threshold=3.0, threshold=2.5, dpi=300, logger=None):
    """Plot overlay map on inflated brain surface with custom legend.

    Args:
        overlay_img (Nifti1Image): Overlay map with three-color coding
        save_path (str): Path to save the image
        title (str): Title for the plot
        legend_img_path (str): Path to legend image
        color1 (str): Hex color for condition 1
        color2 (str): Hex color for condition 2
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

    # Render each view separately at high DPI for quality
    render_dpi = 150

    left_lateral = plot_single_surface_view(overlay_img, 'left', 'lateral', cmap, threshold, render_dpi)
    left_medial = plot_single_surface_view(overlay_img, 'left', 'medial', cmap, threshold, render_dpi)
    right_lateral = plot_single_surface_view(overlay_img, 'right', 'lateral', cmap, threshold, render_dpi)
    right_medial = plot_single_surface_view(overlay_img, 'right', 'medial', cmap, threshold, render_dpi)

    # Convert arrays to PIL Images and crop whitespace
    from PIL import Image, ImageDraw, ImageFont

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
    lm_img = crop_whitespace(left_medial)
    rl_img = crop_whitespace(right_lateral)
    rm_img = crop_whitespace(right_medial)

    # Load legend
    legend_img = Image.open(legend_img_path)

    # Create compact 2x2 grid layout
    # Layout: [Left Lateral | Left Medial]   [Legend]
    #         [Right Lateral | Right Medial] [Legend]
    spacing = 5  # Minimal spacing in pixels

    # Make all brain images the same height
    max_height = max(ll_img.height, lm_img.height, rl_img.height, rm_img.height)

    def resize_to_height(img, target_height):
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, target_height), Image.LANCZOS)

    ll_img = resize_to_height(ll_img, max_height)
    lm_img = resize_to_height(lm_img, max_height)
    rl_img = resize_to_height(rl_img, max_height)
    rm_img = resize_to_height(rm_img, max_height)

    # Calculate dimensions for 2x2 grid
    # Two rows of brain images
    row_width = max(ll_img.width + lm_img.width, rl_img.width + rm_img.width) + spacing
    grid_height = 2 * max_height + spacing

    # Resize legend to match grid height
    legend_img = resize_to_height(legend_img, grid_height)

    # Calculate total dimensions
    total_width = row_width + legend_img.width + 3 * spacing
    title_height = 40
    total_height = grid_height + title_height + spacing

    # Create final composite image
    final_img = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images in 2x2 grid with minimal spacing
    y_offset = title_height

    # Top row: Left Lateral | Left Medial
    x_offset = spacing
    final_img.paste(ll_img, (x_offset, y_offset))
    x_offset += ll_img.width + spacing
    final_img.paste(lm_img, (x_offset, y_offset))

    # Bottom row: Right Lateral | Right Medial
    y_offset += max_height + spacing
    x_offset = spacing
    final_img.paste(rl_img, (x_offset, y_offset))
    x_offset += rl_img.width + spacing
    final_img.paste(rm_img, (x_offset, y_offset))

    # Paste legend on the right side (spanning both rows)
    legend_x = row_width + 2 * spacing
    legend_y = title_height
    final_img.paste(legend_img, (legend_x, legend_y))

    # Add title using PIL
    draw = ImageDraw.Draw(final_img)
    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Center title
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (total_width - title_width) // 2
    draw.text((title_x, 10), title, fill='black', font=font)

    # Save final image
    final_img.save(save_path, dpi=(dpi, dpi))

    if logger:
        logger.debug(f"Saved compact overlay surface plot: {save_path}")


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


def create_legend_image(save_path, source1, cond1, source2, cond2, dpi=150):
    """Create a vertical legend with Blue/Red/Purple gradients and task-colored labels.

    Args:
        save_path (str): Path to save legend image
        source1 (str): Source for condition 1 ('shinobi' or 'hcp')
        cond1 (str): Condition 1 name
        source2 (str): Source for condition 2 ('shinobi' or 'hcp')
        cond2 (str): Condition 2 name
        dpi (int): DPI of output image
    """
    # Get colors for condition labels (task-specific)
    color1 = get_condition_color(source1, cond1)
    color2 = get_condition_color(source2, cond2)

    # Vertical layout with thinner bars
    fig, ax = plt.subplots(figsize=(1.5, 6), dpi=dpi)
    ax.axis('off')

    # Create gradient colorbars - THINNER bars (vertical orientation)
    # Darker at top, lighter at bottom
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)  # Top to bottom: 1 to 0

    # Blue gradient for condition 1
    ax_cond1 = fig.add_axes([0.15, 0.68, 0.25, 0.25])
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cond1', ['#00008B', '#6666FF'])
    ax_cond1.imshow(gradient, aspect='auto', cmap=cmap1)
    ax_cond1.set_xticks([])
    ax_cond1.set_yticks([])
    for spine in ax_cond1.spines.values():
        spine.set_visible(True)

    # Red gradient for condition 2
    ax_cond2 = fig.add_axes([0.15, 0.39, 0.25, 0.25])
    cmap2 = mcolors.LinearSegmentedColormap.from_list('cond2', ['#8B0000', '#FF6666'])
    ax_cond2.imshow(gradient, aspect='auto', cmap=cmap2)
    ax_cond2.set_xticks([])
    ax_cond2.set_yticks([])
    for spine in ax_cond2.spines.values():
        spine.set_visible(True)

    # Purple gradient for both conditions
    ax_both = fig.add_axes([0.15, 0.10, 0.25, 0.25])
    ax_both.imshow(gradient, aspect='auto', cmap=mcolors.LinearSegmentedColormap.from_list('both', ['#6600CC', '#CC66FF']))
    ax_both.set_xticks([])
    ax_both.set_yticks([])
    for spine in ax_both.spines.values():
        spine.set_visible(True)

    # Add labels to the right of each bar - condition names colored by task
    ax.text(0.50, 0.805, cond1, ha='left', va='center', fontsize=11,
            fontweight='bold', color=color1, transform=ax.transAxes)
    ax.text(0.50, 0.515, cond2, ha='left', va='center', fontsize=11,
            fontweight='bold', color=color2, transform=ax.transAxes)
    ax.text(0.50, 0.225, 'Both', ha='left', va='center', fontsize=11,
            fontweight='bold', color='black', transform=ax.transAxes)  # Black for "Both"

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    """Main function to generate condition comparison plots."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--cond1',
        type=str,
        required=True,
        help='First condition in format "source:condition" (e.g., "shinobi:Kill" or "hcp:reward")'
    )
    parser.add_argument(
        '--cond2',
        type=str,
        required=True,
        help='Second condition in format "source:condition" (e.g., "shinobi:HealthLoss" or "hcp:punishment")'
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
        # Parse condition specifications
        source1, cond1 = parse_condition_spec(args.cond1)
        source2, cond2 = parse_condition_spec(args.cond2)

        logger.info(f"Comparing {source1}:{cond1} vs {source2}:{cond2}")
        logger.info(f"Subjects: {', '.join(SUBJECTS)}")
        logger.info(f"Threshold: |z| > {args.threshold}\n")

        # Determine HCP task if needed
        hcp_task1 = None
        hcp_task2 = None
        if source1 == 'hcp':
            hcp_task1 = find_hcp_task(cond1, args.hcp_task)
            logger.info(f"Using HCP task '{hcp_task1}' for condition 1")
        if source2 == 'hcp':
            hcp_task2 = find_hcp_task(cond2, args.hcp_task)
            logger.info(f"Using HCP task '{hcp_task2}' for condition 2")

        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = op.join(
                ".", "reports", "figures", "condition_comparison",
                f"{source1}_{cond1}_vs_{source2}_{cond2}"
            )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}\n")

        # Get task colors for conditions
        color1 = get_condition_color(source1, cond1)
        color2 = get_condition_color(source2, cond2)
        logger.info(f"Using colors: {cond1}={color1}, {cond2}={color2}")

        # Create legend
        legend_path = op.join(output_dir, "legend.png")
        create_legend_image(legend_path, source1, cond1, source2, cond2)
        logger.info(f"Created legend: {legend_path}")

        # Process each subject
        for subject in tqdm(SUBJECTS, desc="Processing subjects"):
            logger.info(f"\nProcessing {subject}")

            # Load z-maps
            img1 = load_zmap(source1, subject, cond1, hcp_task1, args.data_path, logger)
            img2 = load_zmap(source2, subject, cond2, hcp_task2, args.data_path, logger)

            if img1 is None or img2 is None:
                logger.warning(f"Skipping {subject} due to missing data")
                continue

            # Create overlay
            overlay_img = create_overlay_map(img1, img2, args.threshold, logger)

            # Plot overlay on surface with task-specific colors
            save_path = op.join(output_dir, f"{subject}_comparison.png")
            title = f"{subject}"  # Simplified title
            plot_overlay_surface(overlay_img, save_path, title, legend_path, color1, color2,
                               img1=img1, img2=img2, z_threshold=args.threshold,
                               threshold=2.5, logger=logger)

            logger.info(f"Saved: {save_path}")

        # Print summary
        logger.info("\nAll done!")
        logger.info(f"Comparison plots saved to {op.abspath(output_dir)}/")
        logger.info(f"Legend: {op.abspath(legend_path)}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
