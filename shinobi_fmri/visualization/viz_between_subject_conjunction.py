#!/usr/bin/env python
"""
Generate between-subject conjunction maps on inflated brain surfaces.

For each condition/annotation, counts how many of the 4 subjects show activation
at each voxel (using corrected z-maps where non-zero = significant). Produces a
single brain plot per condition with a discrete 4-level sequential colormap
(light to dark) showing 1, 2, 3, or 4 subjects activated.

When processing all conditions, creates a combined panel (3x4 grid) with
all conjunction maps and the legend in the 12th cell.

Usage:
    # Single condition
    python viz_between_subject_conjunction.py --condition Kill -v

    # Multiple conditions
    python viz_between_subject_conjunction.py --conditions Kill,HIT,JUMP -v

    # All conditions (default)
    python viz_between_subject_conjunction.py -v

    # Use raw maps with threshold instead of corrected maps
    python viz_between_subject_conjunction.py --use-raw-maps --threshold 3.0 -v
"""

import os
import os.path as op
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from nilearn import plotting
from PIL import Image
from tqdm import tqdm
from shinobi_fmri.utils.logger import AnalysisLogger
import logging

# Filter specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="numpy",
    message="Warning: 'partition' will ignore the 'mask' of the MaskedArray",
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message="The `darkness` parameter will be deprecated",
)

try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS, CONDITIONS, LOW_LEVEL_CONDITIONS
except ImportError:
    print("Warning: config not found. Using defaults.")
    DATA_PATH = "/home/hyruuk/scratch/data"
    SUBJECTS = ["sub-01", "sub-02", "sub-04", "sub-06"]
    CONDITIONS = ["HIT", "JUMP", "DOWN", "HealthLoss", "Kill", "LEFT", "RIGHT", "UP"]
    LOW_LEVEL_CONDITIONS = ["luminance", "optical_flow", "audio_envelope", "button_presses_count"]

# Default threshold for raw maps
DEFAULT_THRESHOLD = 3.0

# Conjunction colormap colors (4 discrete levels)
CONJUNCTION_COLORS = {
    1: "#FFCCCC",  # light pink/salmon
    2: "#FF6666",  # medium red
    3: "#CC0000",  # dark red
    4: "#660000",  # very dark red/maroon
}


def create_conjunction_map(subjects, condition, data_path, use_corrected_maps=True,
                           threshold=DEFAULT_THRESHOLD, logger=None):
    """Create a conjunction map counting subjects with activation at each voxel.

    For each subject, loads the z-map and creates a binary mask (non-zero for
    corrected maps, or above threshold for raw maps). Sums masks across subjects
    to produce an integer map with values 0-4.

    Args:
        subjects (list): List of subject IDs (e.g., ['sub-01', 'sub-02', ...])
        condition (str): Condition name (e.g., 'Kill')
        data_path (str): Path to data directory
        use_corrected_maps (bool): If True, use corrected z-maps (default: True)
        threshold (float): Threshold for raw maps (only used when use_corrected_maps=False)
        logger (AnalysisLogger): Logger instance

    Returns:
        tuple: (Nifti1Image with values 0-N_subjects, int count of loaded subjects)
            Returns (None, 0) if no subjects could be loaded.
    """
    binary_masks = []
    reference_img = None
    loaded_count = 0

    for subject in subjects:
        img = _load_subject_zmap(subject, condition, data_path, use_corrected_maps, logger)
        if img is None:
            continue

        if reference_img is None:
            reference_img = img

        data = img.get_fdata()

        if use_corrected_maps:
            mask = np.abs(data) > 1e-6
        else:
            mask = np.abs(data) > threshold

        binary_masks.append(mask.astype(np.int32))
        loaded_count += 1

    if loaded_count == 0:
        if logger:
            logger.warning(f"No subjects loaded for condition {condition}")
        return None, 0

    conjunction = np.sum(binary_masks, axis=0)

    if logger:
        for n_subj in range(1, loaded_count + 1):
            n_voxels = np.sum(conjunction >= n_subj)
            logger.info(f"  {n_subj}+ subjects: {n_voxels} voxels")

    conjunction_img = nib.Nifti1Image(
        conjunction.astype(np.float32), reference_img.affine, reference_img.header
    )
    return conjunction_img, loaded_count


def _load_subject_zmap(subject, condition, data_path, use_corrected_maps, logger=None):
    """Load a subject-level z-map, falling back to raw if corrected not found.

    Args:
        subject (str): Subject ID
        condition (str): Condition name
        data_path (str): Path to data directory
        use_corrected_maps (bool): Whether to use corrected maps
        logger (AnalysisLogger): Logger instance

    Returns:
        Nifti1Image or None: Loaded image, or None if not found
    """
    if use_corrected_maps:
        filename = f"{subject}_task-shinobi_contrast-{condition}_desc-corrected_stat-z.nii.gz"
    else:
        filename = f"{subject}_task-shinobi_contrast-{condition}_stat-z.nii.gz"

    path = op.join(data_path, "processed", "subject-level", subject, "z_maps", filename)

    # Fall back to raw map if corrected doesn't exist
    if use_corrected_maps and not op.isfile(path):
        raw_filename = f"{subject}_task-shinobi_contrast-{condition}_stat-z.nii.gz"
        raw_path = op.join(data_path, "processed", "subject-level", subject, "z_maps", raw_filename)
        if op.isfile(raw_path):
            if logger:
                logger.debug(f"Corrected map not found, using raw: {raw_path}")
            path = raw_path

    if not op.isfile(path):
        if logger:
            logger.warning(f"Map not found: {path}")
        return None

    try:
        img = nib.load(path)
        if logger:
            logger.debug(f"Loaded: {path}")
        return img
    except Exception as e:
        if logger:
            logger.error(f"Error loading {path}: {e}")
        return None


def create_conjunction_colormap():
    """Create a 4-level discrete sequential colormap for conjunction maps.

    Returns:
        matplotlib.colors.ListedColormap: 4-color discrete colormap
    """
    colors = [
        CONJUNCTION_COLORS[1],
        CONJUNCTION_COLORS[2],
        CONJUNCTION_COLORS[3],
        CONJUNCTION_COLORS[4],
    ]
    return mcolors.ListedColormap(colors, name="conjunction_4level")


def create_conjunction_legend(dpi=300):
    """Create a vertical legend showing the 4 conjunction levels.

    Args:
        dpi (int): DPI for rendering

    Returns:
        PIL.Image: Legend as PIL Image
    """
    import io

    fig, ax = plt.subplots(figsize=(3, 4), dpi=dpi)
    ax.axis("off")

    labels = ["1 subject", "2 subjects", "3 subjects", "4 subjects"]
    colors = [CONJUNCTION_COLORS[i] for i in range(1, 5)]

    # Draw colored squares with labels
    y_positions = [0.80, 0.60, 0.40, 0.20]
    box_size = 0.08

    for y_pos, color, label in zip(y_positions, colors, labels):
        # Draw colored square
        rect = plt.Rectangle(
            (0.1, y_pos - box_size / 2), box_size * 0.75, box_size,
            facecolor=color, edgecolor="black", linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        # Label text
        ax.text(
            0.25, y_pos, label, ha="left", va="center",
            fontsize=18, transform=ax.transAxes,
        )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    buf.seek(0)
    legend_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    return legend_img


def _render_single_view(img, hemisphere, view, cmap, dpi=300):
    """Render a single surface view and return as PIL Image.

    Args:
        img (Nifti1Image): Volume image to plot
        hemisphere (str): 'left' or 'right'
        view (str): 'lateral' or 'medial'
        cmap: Colormap to use
        dpi (int): DPI for rendering

    Returns:
        PIL.Image: Cropped rendered view
    """
    plt.rcParams["figure.dpi"] = dpi

    plotting.plot_img_on_surf(
        img,
        surf_mesh="fsaverage5",
        views=[view],
        hemispheres=[hemisphere],
        inflate=True,
        colorbar=False,
        threshold=0.5,
        vmin=0.5,
        vmax=4.5,
        symmetric_cbar=False,
        cmap=cmap,
        darkness=None,
    )

    fig = plt.gcf()
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    pil_img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)

    return _crop_whitespace(pil_img)


def _crop_whitespace(img):
    """Crop whitespace from a PIL Image.

    Args:
        img (PIL.Image): Input image

    Returns:
        PIL.Image: Cropped image
    """
    gray = img.convert("L")
    bbox = gray.point(lambda x: 0 if x > 250 else 255).getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def _assemble_brain_grid(rendered_views):
    """Assemble 4 rendered views into a 2x2 brain grid.

    Args:
        rendered_views (list): List of 4 PIL Images [left_lat, right_lat, left_med, right_med]

    Returns:
        PIL.Image: Assembled 2x2 brain grid
    """
    # Resize all to same height
    max_height = max(im.height for im in rendered_views)
    resized = []
    for im in rendered_views:
        if im.height != max_height:
            ratio = max_height / im.height
            new_width = int(im.width * ratio)
            im = im.resize((new_width, max_height), Image.LANCZOS)
        resized.append(im)

    ll, rl, lm, rm = resized

    spacing = 5
    row1_w = ll.width + spacing + rl.width
    row2_w = lm.width + spacing + rm.width
    grid_w = max(row1_w, row2_w)
    grid_h = max_height * 2 + spacing

    grid_img = Image.new("RGB", (grid_w, grid_h), "white")

    # Row 1: Left Lateral | Right Lateral (centered)
    r1_x = (grid_w - row1_w) // 2
    grid_img.paste(ll, (r1_x, 0))
    grid_img.paste(rl, (r1_x + ll.width + spacing, 0))

    # Row 2: Left Medial | Right Medial (centered)
    r2_x = (grid_w - row2_w) // 2
    grid_img.paste(lm, (r2_x, max_height + spacing))
    grid_img.paste(rm, (r2_x + lm.width + spacing, max_height + spacing))

    return grid_img


def _render_title(title, width, dpi=300):
    """Render a condition title as a PIL Image.

    Args:
        title (str): Condition name
        width (int): Target width in pixels
        dpi (int): DPI for rendering

    Returns:
        PIL.Image: Title image
    """
    import io
    from shinobi_fmri.visualization.hcp_tasks import LOW_LEVEL_DISPLAY_NAMES

    display_title = LOW_LEVEL_DISPLAY_NAMES.get(title, title)

    title_height_in = 0.3
    title_fig = plt.figure(figsize=(width / dpi, title_height_in), dpi=dpi)
    title_ax = title_fig.add_subplot(111)
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.8, display_title, ha="center", va="top",
        fontsize=28, fontweight="normal", transform=title_ax.transAxes,
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    buf.seek(0)
    title_pil = Image.open(buf).copy()
    buf.close()
    plt.close(title_fig)

    return title_pil


def _render_conjunction_brain(conjunction_img, title, dpi=300):
    """Render a conjunction brain map with title (no legend).

    Args:
        conjunction_img (Nifti1Image): Conjunction map with values 0-4
        title (str): Condition name for the title
        dpi (int): DPI for rendering

    Returns:
        PIL.Image: Brain grid with title above it
    """
    cmap = create_conjunction_colormap()

    # Render 4 views
    views = [
        ("left", "lateral"),
        ("right", "lateral"),
        ("left", "medial"),
        ("right", "medial"),
    ]
    rendered = [_render_single_view(conjunction_img, h, v, cmap, dpi=300) for h, v in views]

    # Assemble 2x2 brain grid
    brain_grid = _assemble_brain_grid(rendered)

    # Render title
    title_img = _render_title(title, brain_grid.width, dpi)

    # Combine title + brain
    title_gap = 2
    total_h = title_img.height + title_gap + brain_grid.height
    total_w = max(brain_grid.width, title_img.width)
    combined = Image.new("RGB", (total_w, total_h), "white")

    # Center title
    title_x = (total_w - title_img.width) // 2
    combined.paste(title_img, (max(0, title_x), 0))

    # Center brain grid
    brain_x = (total_w - brain_grid.width) // 2
    combined.paste(brain_grid, (max(0, brain_x), title_img.height + title_gap))

    return combined


def plot_conjunction_surface(conjunction_img, save_path, title=None, dpi=300, logger=None):
    """Plot conjunction map on inflated brain surface with title and legend.

    Creates a composite of 4 views (Left/Right x Lateral/Medial) with a title
    and legend showing the discrete colormap. Saves to a PNG file.

    Args:
        conjunction_img (Nifti1Image): Conjunction map with values 0-4
        save_path (str): Path to save the output PNG
        title (str): Title for the figure
        dpi (int): Output DPI
        logger (AnalysisLogger): Logger instance
    """
    if logger:
        logger.debug("Rendering conjunction surface views...")

    # Render brain with title (no legend)
    brain_img = _render_conjunction_brain(conjunction_img, title or "", dpi)

    # Create legend
    legend_img = create_conjunction_legend(dpi)

    # Combine brain + legend side by side
    legend_margin = 40
    total_w = brain_img.width + legend_margin + legend_img.width
    total_h = max(brain_img.height, legend_img.height)

    final_img = Image.new("RGB", (total_w, total_h), "white")
    final_img.paste(brain_img, (0, 0))

    # Center legend vertically
    legend_y = (total_h - legend_img.height) // 2
    final_img.paste(legend_img, (brain_img.width + legend_margin, max(0, legend_y)))

    final_img.save(save_path, dpi=(dpi, dpi))

    if logger:
        logger.info(f"Saved: {save_path}")


def create_conjunction_panel(brain_images, legend_img, save_path, n_cols=3, dpi=300, logger=None):
    """Create a combined panel with all conjunction maps in a grid.

    Arranges brain images in a grid with the legend in the last cell.
    For 11 conditions + 1 legend = 12 cells in a 3x4 grid.

    Args:
        brain_images (list): List of (condition_name, PIL.Image) tuples (titled brain images)
        legend_img (PIL.Image): Legend image for the last cell
        save_path (str): Path to save the panel PNG
        n_cols (int): Number of columns in the grid (default: 3)
        dpi (int): Output DPI
        logger (AnalysisLogger): Logger instance
    """
    n_items = len(brain_images) + 1  # +1 for legend
    n_rows = (n_items + n_cols - 1) // n_cols

    if not brain_images:
        if logger:
            logger.warning("No brain images for panel")
        return

    # Find maximum cell dimensions
    max_cell_w = max(img.width for _, img in brain_images)
    max_cell_h = max(img.height for _, img in brain_images)

    # Grid spacing between cells
    cell_spacing = 120

    total_w = n_cols * max_cell_w + (n_cols - 1) * cell_spacing
    total_h = n_rows * max_cell_h + (n_rows - 1) * cell_spacing

    panel = Image.new("RGB", (total_w, total_h), "white")

    # Place brain images
    for idx, (condition, img) in enumerate(brain_images):
        row = idx // n_cols
        col = idx % n_cols

        x = col * (max_cell_w + cell_spacing)
        y = row * (max_cell_h + cell_spacing)

        # Center the image within the cell
        x_offset = (max_cell_w - img.width) // 2
        y_offset = (max_cell_h - img.height) // 2
        panel.paste(img, (x + x_offset, y + y_offset))

    # Place legend in the last cell, scaled to fill the cell
    legend_idx = len(brain_images)
    legend_row = legend_idx // n_cols
    legend_col = legend_idx % n_cols

    legend_x = legend_col * (max_cell_w + cell_spacing)
    legend_y = legend_row * (max_cell_h + cell_spacing)

    # Scale legend to fill the cell while preserving aspect ratio
    scale_w = max_cell_w / legend_img.width
    scale_h = max_cell_h / legend_img.height
    scale = min(scale_w, scale_h)
    new_legend_w = int(legend_img.width * scale)
    new_legend_h = int(legend_img.height * scale)
    scaled_legend = legend_img.resize((new_legend_w, new_legend_h), Image.LANCZOS)

    # Center scaled legend within the cell
    lx_offset = (max_cell_w - new_legend_w) // 2
    ly_offset = (max_cell_h - new_legend_h) // 2
    panel.paste(scaled_legend, (legend_x + lx_offset, legend_y + ly_offset))

    panel.save(save_path, dpi=(dpi, dpi))

    if logger:
        logger.info(f"Saved panel: {save_path} ({n_cols}x{n_rows} grid, {len(brain_images)} maps + legend)")


def main():
    """Main function to generate between-subject conjunction plots."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--condition",
        type=str, default=None,
        help="Single condition to process (e.g., Kill)",
    )
    parser.add_argument(
        "--conditions",
        type=str, default=None,
        help="Comma-separated list of conditions (e.g., Kill,HIT,JUMP)",
    )
    parser.add_argument(
        "--data-path",
        type=str, default=DATA_PATH,
        help=f"Path to data directory (default: {DATA_PATH})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str, default=None,
        help="Output directory (default: ./reports/figures/between_subject_conjunction/)",
    )
    parser.add_argument(
        "--use-raw-maps",
        action="store_true",
        help="Use raw (uncorrected) z-maps instead of corrected maps (default: use corrected)",
    )
    parser.add_argument(
        "--threshold",
        type=float, default=DEFAULT_THRESHOLD,
        help=f"Threshold for raw maps (only used with --use-raw-maps, default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--exclude-low-level",
        action="store_true",
        help="Exclude low-level features from processing",
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip generating the combined panel image",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images even if they already exist",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count", default=0,
        help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files",
    )

    args = parser.parse_args()

    # Corrected maps are used by default
    use_corrected_maps = not args.use_raw_maps

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="ConjunctionViz",
        log_dir=args.log_dir,
        verbosity=log_level,
    )

    try:
        # Determine conditions
        if args.condition:
            conditions = [args.condition]
        elif args.conditions:
            conditions = [c.strip() for c in args.conditions.split(",")]
        else:
            conditions = list(CONDITIONS)
            if not args.exclude_low_level:
                conditions += list(LOW_LEVEL_CONDITIONS)

        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = op.join(".", "reports", "figures", "between_subject_conjunction")
        os.makedirs(output_dir, exist_ok=True)

        map_type = "corrected" if use_corrected_maps else "raw"

        logger.info(f"Processing {len(conditions)} condition(s)")
        logger.info(f"Subjects: {', '.join(SUBJECTS)}")
        logger.info(f"Using {'corrected' if use_corrected_maps else 'raw'} z-maps")
        if not use_corrected_maps:
            logger.info(f"Threshold: |z| > {args.threshold}")
        logger.info(f"Output directory: {output_dir}\n")

        generated_count = 0
        # Collect titled brain images (without legend) for the panel
        panel_brain_images = []

        for condition in tqdm(conditions, desc="Conditions", unit="cond"):
            save_path = op.join(
                output_dir,
                f"conjunction_{condition}_{map_type}.png",
            )

            logger.info(f"\nProcessing condition: {condition}")

            conjunction_img, n_subjects = create_conjunction_map(
                SUBJECTS, condition, args.data_path,
                use_corrected_maps=use_corrected_maps,
                threshold=args.threshold,
                logger=logger,
            )

            if conjunction_img is None:
                logger.warning(f"Skipping {condition}: no data loaded")
                continue

            logger.info(f"Loaded {n_subjects} subjects for {condition}")

            # Render brain with title (no legend) — used for both individual PNG and panel
            brain_img = _render_conjunction_brain(conjunction_img, condition, dpi=300)
            panel_brain_images.append((condition, brain_img))

            # Save individual PNG (brain + legend)
            if args.force or not op.isfile(save_path):
                legend_img = create_conjunction_legend(dpi=300)

                legend_margin = 40
                total_w = brain_img.width + legend_margin + legend_img.width
                total_h = max(brain_img.height, legend_img.height)

                final_img = Image.new("RGB", (total_w, total_h), "white")
                final_img.paste(brain_img, (0, 0))
                legend_y = (total_h - legend_img.height) // 2
                final_img.paste(legend_img, (brain_img.width + legend_margin, max(0, legend_y)))
                final_img.save(save_path, dpi=(300, 300))

                logger.info(f"Saved: {save_path}")

            generated_count += 1

        # Create combined panel
        if not args.skip_panel and len(panel_brain_images) > 1:
            panel_path = op.join(output_dir, f"between_subject_conjunction_{map_type}_panel.png")
            legend_img = create_conjunction_legend(dpi=300)

            create_conjunction_panel(
                panel_brain_images, legend_img, panel_path,
                n_cols=3, dpi=300, logger=logger,
            )

        logger.info(f"\nAll done! Generated {generated_count} conjunction map(s)")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
