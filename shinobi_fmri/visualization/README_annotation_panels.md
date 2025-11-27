# Annotation Panels Visualization

This module generates comprehensive visualization panels showing brain activation patterns for annotations/conditions across subjects and sessions.

## What It Creates

The `viz_annotation_panels.py` script generates three types of outputs:

1. **Individual inflated brain maps** (PNG files)
   - Subject-level z-maps (1 per subject per condition)
   - Session-level z-maps (1 per session per condition)

2. **Combined annotation panels** (PNG files)
   - 4×9 grid layout showing all subjects
   - For each subject:
     - 1 large subject-level map (left side)
     - 4 smaller session-level maps (right side, top sessions by activation)
   - Colorbar on the right
   - One panel per condition/annotation

3. **PDF compilation**
   - All annotation panels combined into a single PDF
   - One page per condition

## How It Works

### Selection of Top Sessions

For each subject and condition, the script:
1. Loads all session-level z-maps
2. Counts voxels with |z| > 3 (strong activation)
3. Selects the top 4 sessions with most activated voxels
4. Displays these in the panel

This ensures the most representative sessions are shown for each subject.

### Output Structure

```
reports/figures/full_zmap_plot/
├── sub-01/
│   ├── HIT/
│   │   ├── sub-01_HIT.png              # Subject-level
│   │   ├── sub-01_ses-001_HIT.png      # Session-level
│   │   ├── sub-01_ses-002_HIT.png
│   │   └── ...
│   ├── JUMP/
│   └── ...
├── sub-02/
│   └── ...
└── annotations/
    ├── annotations_plot_HIT.png         # Combined panel
    ├── annotations_plot_JUMP.png
    ├── annotations_plot_Kill.png
    └── inflated_zmaps_by_annot.pdf     # All panels in PDF
```

## Usage

### Via Invoke (Recommended)

```bash
# Process all default conditions
invoke viz.annotation-panels

# Single condition
invoke viz.annotation-panels --condition HIT

# Multiple specific conditions
invoke viz.annotation-panels --conditions "HIT,JUMP,Kill"

# Skip PDF generation (faster for testing)
invoke viz.annotation-panels --skip-pdf

# Only create panels from existing individual images
invoke viz.annotation-panels --skip-individual

# Only create individual images (no panels)
invoke viz.annotation-panels --skip-panels
```

### Direct Script Execution

```bash
python shinobi_fmri/visualization/viz_annotation_panels.py --help

# Examples
python shinobi_fmri/visualization/viz_annotation_panels.py
python shinobi_fmri/visualization/viz_annotation_panels.py --condition HIT
python shinobi_fmri/visualization/viz_annotation_panels.py --output-dir ./custom_output
```

## Configuration

### Default Conditions

```python
DEFAULT_CONDITIONS = ['Kill', 'HealthLoss', 'JUMP', 'HIT', 'DOWN', 'LEFT', 'RIGHT', 'UP']
```

### Default Subjects

Loaded from `shinobi_behav.SUBJECTS`:
```python
SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
```

### Visualization Parameters

- **Colormap**: Nilearn 'cold_hot' with grey zone for |z| < 3
- **Threshold**: z = 3 (voxels below threshold not shown)
- **vmax**: 6 (colormap maximum)
- **DPI**: 300 (high resolution)

## File Requirements

The script expects z-maps at:

```
{DATA_PATH}/processed/z_maps/
├── subject-level/
│   └── {condition}/
│       └── {subject}_simplemodel_{condition}.nii.gz
└── ses-level/
    └── {condition}/
        └── {subject}_{session}_simplemodel_{condition}.nii.gz
```

## Workflow Integration

Typical workflow for generating annotation panels:

```bash
# 1. Run subject-level GLMs (if not done)
invoke batch.glm-subject-level --slurm

# 2. Run session-level GLMs (if not done)
invoke batch.glm-session-level --slurm

# 3. Generate annotation panels
invoke viz.annotation-panels
```

## Performance Notes

- **Individual image generation**: ~5-10 seconds per image (depends on z-map size)
- **Panel creation**: ~30-60 seconds per condition
- **PDF creation**: ~10-20 seconds
- **Full run (8 conditions)**: ~15-30 minutes for all subjects and sessions

Use `--skip-individual` if you've already generated the individual images to save time.

## Troubleshooting

### Missing z-maps

If a z-map file is missing, the script creates a blank white image as a placeholder and continues processing.

### Memory Issues

If you encounter memory errors with many sessions:
- Process conditions one at a time: `--condition HIT`
- Use `--skip-pdf` to avoid loading all panels at once

### Colorbar Not Showing

The colorbar uses a custom colormap. If it's not visible:
- Check that matplotlib and nilearn are properly installed
- Verify DPI settings in your environment

## Code Organization

### Main Functions

- `create_colormap()`: Creates custom grey-centered colormap
- `plot_inflated_zmap()`: Plots single brain map on inflated surface
- `create_all_images()`: Generates individual images for one subject/condition
- `make_annotation_plot()`: Assembles panel for one condition
- `create_pdf_with_images()`: Compiles panels into PDF

### Original Source

This script consolidates and refactors code from:
- `viz_sub-ses_plot.py` (original implementation)
- `viz_subject-level.py` (subject-level visualization utilities)
- `viz_session-level.py` (session-level visualization utilities)

## Related Documentation

- [Task Automation Guide](../../TASKS_USAGE.md) - Using invoke tasks
- [GLM Pipeline](../glm/README.md) - Generating the z-maps
- [Visualization Overview](./README.md) - Other visualization tools
