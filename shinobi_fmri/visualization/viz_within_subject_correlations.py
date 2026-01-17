"""
Within-subject condition correlation analysis and visualization.

Computes and visualizes how maps from different conditions correlate with each other
within individuals, showing condition specificity.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys

# Add parent directory to path for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH, FIG_PATH
from shinobi_fmri.visualization.hcp_tasks import (
    get_condition_label, get_condition_color, is_shinobi_condition,
    get_all_shinobi_conditions
)


def load_correlation_data(correlation_file: Path):
    """Load the correlation matrix and metadata."""
    print(f"Loading correlation data from {correlation_file}...")
    with open(correlation_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['mapnames'])} maps")
    return data


def compute_within_subject_condition_correlations(corr_data):
    """
    For each subject, compute average correlations between each pair of conditions.

    Returns
    -------
    dict
        Keys are subject IDs, values are DataFrames with condition x condition correlations
    """
    corr_matrix = corr_data['corr_matrix']
    subjects = np.array(corr_data['subj'])
    conditions = np.array(corr_data['cond'])
    sources = np.array(corr_data['source'])

    # Get unique subjects and conditions
    unique_subjects = sorted(set(subjects))
    unique_conditions = sorted(set(conditions))

    print(f"\nFound {len(unique_subjects)} subjects and {len(unique_conditions)} conditions")

    results = {}

    for subject in unique_subjects:
        print(f"\nProcessing {subject}...")

        # Get indices for this subject
        subj_mask = subjects == subject
        subj_indices = np.where(subj_mask)[0]

        # Get conditions for this subject
        subj_conditions = conditions[subj_mask]
        subj_sources = sources[subj_mask]

        # Count maps per condition
        condition_counts = pd.Series(subj_conditions).value_counts()
        print(f"  Maps per condition: min={condition_counts.min()}, max={condition_counts.max()}, mean={condition_counts.mean():.1f}")

        # Initialize correlation matrix for conditions
        cond_corr_matrix = np.zeros((len(unique_conditions), len(unique_conditions)))
        cond_corr_counts = np.zeros((len(unique_conditions), len(unique_conditions)))

        # For each pair of conditions
        for i, cond1 in enumerate(unique_conditions):
            for j, cond2 in enumerate(unique_conditions):
                # Get all maps for each condition
                cond1_indices = subj_indices[subj_conditions == cond1]
                cond2_indices = subj_indices[subj_conditions == cond2]

                if len(cond1_indices) == 0 or len(cond2_indices) == 0:
                    cond_corr_matrix[i, j] = np.nan
                    continue

                # Get all pairwise correlations between these conditions
                corr_values = []
                for idx1 in cond1_indices:
                    for idx2 in cond2_indices:
                        if i == j and idx1 == idx2:
                            # Skip self-correlations
                            continue

                        corr_val = corr_matrix[idx1, idx2]
                        if not np.isnan(corr_val):
                            corr_values.append(corr_val)

                if len(corr_values) > 0:
                    cond_corr_matrix[i, j] = np.mean(corr_values)
                    cond_corr_counts[i, j] = len(corr_values)
                else:
                    cond_corr_matrix[i, j] = np.nan

        # Create DataFrame
        df = pd.DataFrame(
            cond_corr_matrix,
            index=unique_conditions,
            columns=unique_conditions
        )

        results[subject] = {
            'correlation_matrix': df,
            'n_comparisons': cond_corr_counts,
            'n_maps_per_condition': condition_counts
        }

        # Print summary statistics
        # Get off-diagonal values (between different conditions)
        off_diag_mask = ~np.eye(len(unique_conditions), dtype=bool)
        off_diag_values = cond_corr_matrix[off_diag_mask]
        off_diag_values = off_diag_values[~np.isnan(off_diag_values)]

        # Get diagonal values (same condition)
        diag_values = np.diag(cond_corr_matrix)
        diag_values = diag_values[~np.isnan(diag_values)]

        print(f"  Same condition: mean={np.mean(diag_values):.3f}, std={np.std(diag_values):.3f}")
        print(f"  Different conditions: mean={np.mean(off_diag_values):.3f}, std={np.std(off_diag_values):.3f}")

    return results, unique_conditions


def aggregate_across_subjects(subject_results, unique_conditions):
    """
    Average the condition correlation matrices across subjects.
    """
    print("\n\nAggregating across subjects...")

    # Stack all correlation matrices
    all_matrices = []
    for subject, data in subject_results.items():
        all_matrices.append(data['correlation_matrix'].values)

    # Average across subjects
    avg_matrix = np.nanmean(all_matrices, axis=0)

    # Create DataFrame
    avg_df = pd.DataFrame(
        avg_matrix,
        index=unique_conditions,
        columns=unique_conditions
    )

    # Print summary
    off_diag_mask = ~np.eye(len(unique_conditions), dtype=bool)
    off_diag_values = avg_matrix[off_diag_mask]
    off_diag_values = off_diag_values[~np.isnan(off_diag_values)]

    diag_values = np.diag(avg_matrix)
    diag_values = diag_values[~np.isnan(diag_values)]

    print(f"Average across subjects:")
    print(f"  Same condition: mean={np.mean(diag_values):.3f}, std={np.std(diag_values):.3f}")
    print(f"  Different conditions: mean={np.mean(off_diag_values):.3f}, std={np.std(off_diag_values):.3f}")
    print(f"  Difference: {np.mean(diag_values) - np.mean(off_diag_values):.3f}")

    return avg_df


def separate_shinobi_hcp_conditions(conditions, include_low_level=False):
    """
    Separate Shinobi and HCP conditions, grouping HCP by task.

    Parameters
    ----------
    conditions : list
        List of all condition names
    include_low_level : bool
        If True, include low-level features as Shinobi conditions

    Returns
    -------
    tuple
        (shinobi_conditions, hcp_conditions)
    """
    from shinobi_fmri.visualization.hcp_tasks import EVENTS_TASK_DICT, get_event_to_task_mapping

    # Get all Shinobi conditions (including low-level if specified)
    all_shinobi = get_all_shinobi_conditions(include_low_level=include_low_level)

    # Filter Shinobi conditions that are in the data
    shinobi_conds = [c for c in all_shinobi if c in conditions]

    # Group HCP conditions by task
    event_to_task = get_event_to_task_mapping()
    hcp_conds = []

    # Task order for grouping
    task_order = ['Gambling', 'Motor', 'Language', 'Social', 'Relational', 'Emotion']

    for task in task_order:
        # Get conditions for this task that are in the data
        task_conditions = EVENTS_TASK_DICT[task]
        for cond in task_conditions:
            if cond in conditions:
                hcp_conds.append(cond)

    # Add any remaining HCP conditions not in the defined tasks
    all_defined = shinobi_conds + hcp_conds
    remaining = [c for c in conditions if c not in all_defined]
    hcp_conds.extend(sorted(remaining))

    return shinobi_conds, hcp_conds


def color_condition_labels(ax, conditions, axis='both', include_low_level=False):
    """
    Color tick labels based on condition type (Shinobi or HCP task).

    Parameters
    ----------
    ax : matplotlib axis
        The axis to modify
    conditions : list
        List of condition names (without icons) in order
    axis : str
        Which axis to color: 'x', 'y', or 'both'
    include_low_level : bool
        If True, consider low-level features as Shinobi conditions
    """
    if axis in ['x', 'both']:
        for i, (label, cond) in enumerate(zip(ax.get_xticklabels(), conditions)):
            color = get_condition_color(cond)
            label.set_color(color)
            if is_shinobi_condition(cond, include_low_level=include_low_level):
                label.set_fontweight('bold')

    if axis in ['y', 'both']:
        for i, (label, cond) in enumerate(zip(ax.get_yticklabels(), conditions)):
            color = get_condition_color(cond)
            label.set_color(color)
            if is_shinobi_condition(cond, include_low_level=include_low_level):
                label.set_fontweight('bold')


def plot_subject_heatmap(corr_matrix, subject, output_path, include_low_level=False):
    """
    Plot heatmap for a single subject with Shinobi/HCP separation.

    Parameters
    ----------
    corr_matrix : DataFrame
        Correlation matrix
    subject : str
        Subject ID
    output_path : Path
        Output directory
    include_low_level : bool
        If True, include low-level features as Shinobi conditions
    """
    # Reorder conditions: Shinobi first, then HCP
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(
        corr_matrix.index.tolist(),
        include_low_level=include_low_level
    )
    ordered_conds = shinobi_conds + hcp_conds

    # Add icons to condition labels
    ordered_conds_with_icons = [get_condition_label(cond) for cond in ordered_conds]

    # Reorder matrix
    corr_matrix_ordered = corr_matrix.loc[ordered_conds, ordered_conds]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot heatmap
    sns.heatmap(
        corr_matrix_ordered,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.8,
        square=True,
        cbar_kws={'label': 'Correlation (Pearson r)', 'shrink': 0.8},
        ax=ax,
        annot_kws={'fontsize': 7},
        xticklabels=ordered_conds_with_icons,
        yticklabels=ordered_conds_with_icons
    )

    # Color tick labels
    color_condition_labels(ax, ordered_conds, axis='both', include_low_level=include_low_level)

    # Add separating lines between Shinobi and HCP, and between HCP tasks
    from shinobi_fmri.visualization.hcp_tasks import EVENTS_TASK_DICT, get_event_to_task_mapping

    n_shinobi = len(shinobi_conds)
    if n_shinobi > 0 and len(hcp_conds) > 0:
        # Main separator between Shinobi and HCP
        ax.axhline(y=n_shinobi, color='black', linewidth=3)
        ax.axvline(x=n_shinobi, color='black', linewidth=3)

        # Add task separators within HCP section
        event_to_task = get_event_to_task_mapping()
        current_pos = n_shinobi
        prev_task = None

        for i, cond in enumerate(hcp_conds):
            task = event_to_task.get(cond)
            if task and task != prev_task and prev_task is not None:
                # Add a thin line between different tasks
                pos = n_shinobi + i
                ax.axhline(y=pos, color='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.axvline(x=pos, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            prev_task = task

        # Add labels for sections
        ax.text(n_shinobi/2, -0.2, 'Shinobi', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#FF6B6B')
        ax.text(n_shinobi + len(hcp_conds)/2, -0.2, 'HCP', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#4ECDC4')

    ax.set_title(f'{subject}: Within-Subject Condition Correlations\n',
                 fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path / f'{subject}_condition_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / f'{subject}_condition_correlations.png'}")
    plt.close()


def plot_average_heatmap(avg_matrix, output_path, include_low_level=False):
    """
    Plot average heatmap across subjects.

    Parameters
    ----------
    avg_matrix : DataFrame
        Average correlation matrix
    output_path : Path
        Output directory
    include_low_level : bool
        If True, include low-level features as Shinobi conditions
    """
    # Reorder conditions
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(
        avg_matrix.index.tolist(),
        include_low_level=include_low_level
    )
    ordered_conds = shinobi_conds + hcp_conds

    # Add icons to condition labels
    ordered_conds_with_icons = [get_condition_label(cond) for cond in ordered_conds]

    # Reorder matrix
    avg_matrix_ordered = avg_matrix.loc[ordered_conds, ordered_conds]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot heatmap
    sns.heatmap(
        avg_matrix_ordered,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.8,
        square=True,
        cbar_kws={'label': 'Correlation (Pearson r)', 'shrink': 0.8},
        ax=ax,
        annot_kws={'fontsize': 7},
        xticklabels=ordered_conds_with_icons,
        yticklabels=ordered_conds_with_icons
    )

    # Color tick labels
    color_condition_labels(ax, ordered_conds, axis='both', include_low_level=include_low_level)

    # Add separating lines between Shinobi and HCP, and between HCP tasks
    from shinobi_fmri.visualization.hcp_tasks import EVENTS_TASK_DICT, get_event_to_task_mapping

    n_shinobi = len(shinobi_conds)
    if n_shinobi > 0 and len(hcp_conds) > 0:
        # Main separator between Shinobi and HCP
        ax.axhline(y=n_shinobi, color='black', linewidth=3)
        ax.axvline(x=n_shinobi, color='black', linewidth=3)

        # Add task separators within HCP section
        event_to_task = get_event_to_task_mapping()
        prev_task = None

        for i, cond in enumerate(hcp_conds):
            task = event_to_task.get(cond)
            if task and task != prev_task and prev_task is not None:
                # Add a thin line between different tasks
                pos = n_shinobi + i
                ax.axhline(y=pos, color='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.axvline(x=pos, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            prev_task = task

        # Add labels
        ax.text(n_shinobi/2, -0.2, 'Shinobi', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#FF6B6B')
        ax.text(n_shinobi + len(hcp_conds)/2, -0.2, 'HCP', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#4ECDC4')

    ax.set_title('Average Within-Subject Correlations\n',
                 fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path / 'average_condition_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'average_condition_correlations.png'}")
    plt.close()


def plot_same_vs_different_comparison(subject_results, corr_data, output_path, include_low_level=False):
    """
    Plot comparison of same-condition vs different-condition correlations (left),
    and same-subject vs different-subjects correlations by task (right).

    Parameters
    ----------
    subject_results : dict
        Within-subject correlation results
    corr_data : dict
        Raw correlation data
    output_path : Path
        Output directory
    include_low_level : bool
        If True, include low-level features as Shinobi conditions
    """
    # Left panel data: same vs different conditions (within-subject)
    data_for_plot_left = []

    for subject, results in subject_results.items():
        corr_matrix = results['correlation_matrix'].values

        # Same condition (diagonal)
        diag_values = np.diag(corr_matrix)
        diag_values = diag_values[~np.isnan(diag_values)]

        # Different conditions (off-diagonal)
        off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag_values = corr_matrix[off_diag_mask]
        off_diag_values = off_diag_values[~np.isnan(off_diag_values)]

        # Add to data
        for val in diag_values:
            data_for_plot_left.append({
                'subject': subject,
                'comparison_type': 'Same condition',
                'correlation': val
            })

        for val in off_diag_values:
            data_for_plot_left.append({
                'subject': subject,
                'comparison_type': 'Different conditions',
                'correlation': val
            })

    df_left = pd.DataFrame(data_for_plot_left)

    # Right panel data: same-subject vs different-subjects by task
    from shinobi_fmri.visualization.hcp_tasks import get_event_to_task_mapping, EVENTS_TASK_DICT

    data_for_plot_right = []
    corr_matrix = corr_data['corr_matrix']
    subjects = np.array(corr_data['subj'])
    conditions = np.array(corr_data['cond'])
    sources = np.array(corr_data['source'])

    # Map each condition to its task
    event_to_task = get_event_to_task_mapping()

    # Define tasks to analyze
    shinobi_conditions = get_all_shinobi_conditions(include_low_level=include_low_level)
    tasks_to_analyze = ['Shinobi'] + list(EVENTS_TASK_DICT.keys())

    for task in tasks_to_analyze:
        # Get indices for this task
        if task == 'Shinobi':
            task_mask = np.isin(conditions, shinobi_conditions)
        else:
            task_conditions = EVENTS_TASK_DICT[task]
            task_mask = np.isin(conditions, task_conditions)

        task_indices = np.where(task_mask)[0]

        if len(task_indices) < 2:
            continue

        # For each pair of maps in this task
        for i in range(len(task_indices)):
            for j in range(i + 1, len(task_indices)):
                idx1 = task_indices[i]
                idx2 = task_indices[j]

                corr_val = corr_matrix[idx1, idx2]
                if np.isnan(corr_val):
                    continue

                # Check if same subject
                is_same_subject = subjects[idx1] == subjects[idx2]

                data_for_plot_right.append({
                    'task': task,
                    'comparison_type': 'Same subject' if is_same_subject else 'Different subjects',
                    'correlation': corr_val
                })

    df_right = pd.DataFrame(data_for_plot_right)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left panel: Violin plot - same vs different conditions
    ax = axes[0]
    sns.violinplot(data=df_left, x='subject', y='correlation', hue='comparison_type',
                   palette=['#FF6B6B', '#4ECDC4'], ax=ax, split=False)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_title('Same vs Different Condition Correlations by Subject',
                 fontsize=16)
    ax.legend(title='Comparison Type', fontsize=10)
    ax.grid(alpha=0.3, axis='y', linestyle='--')

    # Right panel: Violin plot - same-subject vs different-subjects by task
    ax = axes[1]

    # Order tasks: Shinobi first, then HCP tasks
    task_order = ['Shinobi'] + [t for t in EVENTS_TASK_DICT.keys() if t in df_right['task'].unique()]

    # Create task-specific colors with saturation offset for same vs different subjects
    from shinobi_fmri.visualization.hcp_tasks import get_task_color, get_task_icon, SHINOBI_COLOR, get_task_label
    import colorsys

    def offset_saturation(rgb_color, offset=0.15):
        """Offset the saturation of an RGB color (seaborn colors are RGB tuples)."""
        h, l, s = colorsys.rgb_to_hls(*rgb_color)
        s = max(0.0, min(1.0, s + offset))  # Clamp between 0 and 1
        return colorsys.hls_to_rgb(h, l, s)

    # Create a combined grouping variable for each task-comparison combination
    df_right['task_comparison'] = df_right['task'] + '_' + df_right['comparison_type']

    # Build order for the combined variable
    combined_order = []
    for task in task_order:
        combined_order.append(f'{task}_Same subject')
        combined_order.append(f'{task}_Different subjects')

    # Build color palette mapping for each task-comparison combination
    palette_dict = {}
    for task in task_order:
        if task == 'Shinobi':
            base_color = SHINOBI_COLOR
        else:
            base_color = get_task_color(task)

        # Same subject: lower saturation (desaturate)
        same_color = offset_saturation(base_color, -0.15)
        # Different subjects: higher saturation (saturate)
        diff_color = offset_saturation(base_color, 0.15)

        palette_dict[f'{task}_Same subject'] = same_color
        palette_dict[f'{task}_Different subjects'] = diff_color

    # Plot violin plot using the combined variable
    sns.violinplot(data=df_right, x='task_comparison', y='correlation',
                   order=combined_order, palette=palette_dict, ax=ax, inner='quartile')

    # Create custom x-tick labels with task names and icons (one label per task, centered between its two violins)
    # We'll set labels at positions 0.5, 2.5, 4.5, etc. (middle of each task's pair)
    task_positions = [i * 2 + 0.5 for i in range(len(task_order))]
    task_labels = []
    task_colors_labels = []
    for task in task_order:
        if task == 'Shinobi':
            task_labels.append('Shinobi')
            task_colors_labels.append(SHINOBI_COLOR)
        else:
            icon = get_task_icon(task)
            task_labels.append(f'{icon} {task}')
            task_colors_labels.append(get_task_color(task))

    ax.set_xticks(task_positions)
    ax.set_xticklabels(task_labels, rotation=45, ha='right')

    # Color the x-tick labels with task colors
    for label, color, task in zip(ax.get_xticklabels(), task_colors_labels, task_order):
        label.set_color(color)
        if task == 'Shinobi':
            label.set_fontweight('bold')

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_title('Same-Subject vs Different-Subjects Correlations by Task',
                 fontsize=16)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.5, label='Same subject'),
        Patch(facecolor='gray', alpha=1.0, label='Different subjects')
    ]
    ax.legend(handles=legend_elements, title='Comparison Type', fontsize=10)

    ax.grid(alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / 'same_vs_different_conditions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'same_vs_different_conditions.png'}")
    plt.close()

    # Print statistics for left panel
    print("\n--- Same vs Different Condition Statistics ---")
    same_cond = df_left[df_left['comparison_type'] == 'Same condition']['correlation']
    diff_cond = df_left[df_left['comparison_type'] == 'Different conditions']['correlation']
    print(f"Same condition: mean={same_cond.mean():.3f}, median={same_cond.median():.3f}, std={same_cond.std():.3f}")
    print(f"Different conditions: mean={diff_cond.mean():.3f}, median={diff_cond.median():.3f}, std={diff_cond.std():.3f}")
    print(f"Difference in means: {same_cond.mean() - diff_cond.mean():.3f}")

    from scipy import stats
    cohens_d = (same_cond.mean() - diff_cond.mean()) / np.sqrt((same_cond.std()**2 + diff_cond.std()**2) / 2)
    print(f"Cohen's d: {cohens_d:.3f}")

    t_stat, p_val = stats.ttest_ind(same_cond, diff_cond)
    print(f"t-test: t={t_stat:.3f}, p={p_val:.2e}")

    # Print statistics for right panel
    print("\n--- Same-Subject vs Different-Subjects Statistics by Task ---")
    for task in task_order:
        task_data = df_right[df_right['task'] == task]
        if len(task_data) == 0:
            continue

        same_subj = task_data[task_data['comparison_type'] == 'Same subject']['correlation']
        diff_subj = task_data[task_data['comparison_type'] == 'Different subjects']['correlation']

        if len(same_subj) > 0 and len(diff_subj) > 0:
            print(f"\n{task}:")
            print(f"  Same subject: mean={same_subj.mean():.3f}, median={same_subj.median():.3f}, n={len(same_subj)}")
            print(f"  Different subjects: mean={diff_subj.mean():.3f}, median={diff_subj.median():.3f}, n={len(diff_subj)}")
            print(f"  Difference: {same_subj.mean() - diff_subj.mean():.3f}")


def plot_condition_specificity_matrix(subject_results, output_path, include_low_level=False):
    """
    Plot a matrix showing condition specificity for each subject.
    Specificity = (same-condition correlation) - (mean different-condition correlation)

    Parameters
    ----------
    subject_results : dict
        Within-subject correlation results
    output_path : Path
        Output directory
    include_low_level : bool
        If True, include low-level features as Shinobi conditions
    """
    all_conditions = list(subject_results[list(subject_results.keys())[0]]['correlation_matrix'].index)

    # Separate Shinobi and HCP
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(
        all_conditions,
        include_low_level=include_low_level
    )
    ordered_conds = shinobi_conds + hcp_conds

    # Add icons to condition labels
    ordered_conds_with_icons = [get_condition_label(cond) for cond in ordered_conds]

    subjects = sorted(subject_results.keys())

    specificity_matrix = np.zeros((len(ordered_conds), len(subjects)))

    for j, subject in enumerate(subjects):
        corr_matrix = subject_results[subject]['correlation_matrix']

        for i, condition in enumerate(ordered_conds):
            if condition not in corr_matrix.index:
                specificity_matrix[i, j] = np.nan
                continue

            # Same condition correlation (mean of diagonal for this condition)
            same_cond_corr = corr_matrix.loc[condition, condition]

            # Different condition correlations (mean of row/column excluding diagonal)
            diff_cond_corrs = []
            for other_cond in corr_matrix.index:
                if other_cond != condition:
                    val = corr_matrix.loc[condition, other_cond]
                    if not np.isnan(val):
                        diff_cond_corrs.append(val)

            if len(diff_cond_corrs) > 0:
                mean_diff_cond = np.mean(diff_cond_corrs)
                specificity = same_cond_corr - mean_diff_cond
                specificity_matrix[i, j] = specificity
            else:
                specificity_matrix[i, j] = np.nan

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 12))

    sns.heatmap(
        specificity_matrix,
        xticklabels=subjects,
        yticklabels=ordered_conds_with_icons,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.2,
        vmin=0,
        vmax=0.6,
        cbar_kws={'label': 'Specificity\n(Same - Different)'},
        ax=ax,
        annot_kws={'fontsize': 8}
    )

    # Color tick labels
    color_condition_labels(ax, ordered_conds, axis='y', include_low_level=include_low_level)

    # Add separating lines between Shinobi and HCP, and between HCP tasks
    from shinobi_fmri.visualization.hcp_tasks import get_event_to_task_mapping

    n_shinobi = len(shinobi_conds)
    if n_shinobi > 0 and len(hcp_conds) > 0:
        # Main separator between Shinobi and HCP
        ax.axhline(y=n_shinobi, color='black', linewidth=3)

        # Add task separators within HCP section
        event_to_task = get_event_to_task_mapping()
        prev_task = None

        for i, cond in enumerate(hcp_conds):
            task = event_to_task.get(cond)
            if task and task != prev_task and prev_task is not None:
                # Add a thin line between different tasks
                pos = n_shinobi + i
                ax.axhline(y=pos, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            prev_task = task

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    ax.set_title('Condition Specificity by Subject\n(Same-condition correlation minus mean different-condition correlation)',
                 fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path / 'condition_specificity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'condition_specificity_matrix.png'}")
    plt.close()


def main():
    """Compute correlations and generate all visualizations."""
    parser = argparse.ArgumentParser(description="Within-subject condition correlation analysis")
    parser.add_argument(
        "--low-level-confs",
        action="store_true",
        help="Use correlation data from processed_low-level/ directory"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files",
    )
    args = parser.parse_args()

    print("="*60)
    print("WITHIN-SUBJECT CONDITION CORRELATION ANALYSIS")
    print("="*60)

    # Paths
    processed_dir_name = "processed_low-level" if args.low_level_confs else "processed"
    processed_dir = Path(DATA_PATH) / processed_dir_name
    correlation_file = processed_dir / 'beta_maps_correlations.pkl'

    if args.low_level_confs:
        print(f"\nUsing low-level confounds data from {processed_dir_name}/")
    else:
        print(f"\nUsing standard data from {processed_dir_name}/")

    # Load data
    print("\nLoading correlation data...")
    corr_data = load_correlation_data(correlation_file)

    # Compute within-subject condition correlations
    print("\n" + "="*60)
    print("Computing within-subject condition correlations...")
    print("="*60)
    subject_results, unique_conditions = compute_within_subject_condition_correlations(corr_data)

    # Aggregate across subjects
    avg_matrix = aggregate_across_subjects(subject_results, unique_conditions)

    # Setup output directory
    output_path = Path(FIG_PATH) / 'within_subject_condition_correlations'
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"\nFigure output directory: {output_path}")

    # Generate plots
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # Individual subject heatmaps
    print("\n1. Individual subject heatmaps...")
    for subject in sorted(subject_results.keys()):
        plot_subject_heatmap(
            subject_results[subject]['correlation_matrix'],
            subject,
            output_path,
            include_low_level=args.low_level_confs
        )

    # Average heatmap
    print("\n2. Average heatmap...")
    plot_average_heatmap(avg_matrix, output_path, include_low_level=args.low_level_confs)

    # Same vs different comparison
    print("\n3. Same vs different condition comparison...")
    plot_same_vs_different_comparison(subject_results, corr_data, output_path, include_low_level=args.low_level_confs)

    # Condition specificity matrix
    print("\n4. Condition specificity matrix...")
    plot_condition_specificity_matrix(subject_results, output_path, include_low_level=args.low_level_confs)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_path}")


if __name__ == '__main__':
    main()
