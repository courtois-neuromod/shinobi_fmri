"""
Visualizations for within-subject condition correlations.

Shows how different conditions correlate with each other within each subject.
"""

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
    get_condition_label, get_condition_color, is_shinobi_condition
)


def load_data():
    """Load within-subject condition correlation results."""
    processed_dir = Path(DATA_PATH) / 'processed'
    pickle_file = processed_dir / 'within_subject_condition_correlations' / 'within_subject_condition_correlations.pkl'

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data


def separate_shinobi_hcp_conditions(conditions):
    """Separate Shinobi and HCP conditions, grouping HCP by task."""
    from shinobi_fmri.visualization.hcp_tasks import EVENTS_TASK_DICT, get_event_to_task_mapping

    shinobi_conds = ['DOWN', 'HIT', 'JUMP', 'LEFT', 'RIGHT', 'UP', 'HealthLoss', 'Kill']

    # Filter Shinobi conditions that are in the data
    shinobi_conds = [c for c in shinobi_conds if c in conditions]

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


def color_condition_labels(ax, conditions, axis='both'):
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
    """
    if axis in ['x', 'both']:
        for i, (label, cond) in enumerate(zip(ax.get_xticklabels(), conditions)):
            color = get_condition_color(cond)
            label.set_color(color)
            if is_shinobi_condition(cond):
                label.set_fontweight('bold')

    if axis in ['y', 'both']:
        for i, (label, cond) in enumerate(zip(ax.get_yticklabels(), conditions)):
            color = get_condition_color(cond)
            label.set_color(color)
            if is_shinobi_condition(cond):
                label.set_fontweight('bold')


def plot_subject_heatmap(corr_matrix, subject, output_path):
    """Plot heatmap for a single subject with Shinobi/HCP separation."""

    # Reorder conditions: Shinobi first, then HCP
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(corr_matrix.index.tolist())
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
    color_condition_labels(ax, ordered_conds, axis='both')

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

    ax.set_title(f'{subject}: Within-Subject Condition Correlations\n(How different conditions relate to each other)',
                 fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path / f'{subject}_condition_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / f'{subject}_condition_correlations.png'}")
    plt.close()


def plot_average_heatmap(avg_matrix, output_path):
    """Plot average heatmap across subjects."""

    # Reorder conditions
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(avg_matrix.index.tolist())
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
    color_condition_labels(ax, ordered_conds, axis='both')

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


def plot_same_vs_different_comparison(subject_results, output_path):
    """
    Plot comparison of same-condition vs different-condition correlations.
    """
    data_for_plot = []

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
            data_for_plot.append({
                'subject': subject,
                'comparison_type': 'Same condition',
                'correlation': val
            })

        for val in off_diag_values:
            data_for_plot.append({
                'subject': subject,
                'comparison_type': 'Different conditions',
                'correlation': val
            })

    df = pd.DataFrame(data_for_plot)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Violin plot
    ax = axes[0]
    sns.violinplot(data=df, x='subject', y='correlation', hue='comparison_type',
                   palette=['#FF6B6B', '#4ECDC4'], ax=ax, split=False)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_title('Same vs Different Condition Correlations by Subject',
                 fontsize=16)
    ax.legend(title='Comparison Type', fontsize=10)
    ax.grid(alpha=0.3, axis='y', linestyle='--')

    # Box plot with all subjects combined
    ax = axes[1]
    sns.boxplot(data=df, x='comparison_type', y='correlation',
                palette=['#FF6B6B', '#4ECDC4'], ax=ax)

    # Add individual points
    sns.stripplot(data=df, x='comparison_type', y='correlation',
                  color='black', alpha=0.1, size=2, ax=ax)

    # Calculate and display statistics
    same_cond = df[df['comparison_type'] == 'Same condition']['correlation']
    diff_cond = df[df['comparison_type'] == 'Different conditions']['correlation']

    ax.text(0, ax.get_ylim()[1] * 0.95,
            f'Mean: {same_cond.mean():.3f}\nMedian: {same_cond.median():.3f}',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))

    ax.text(1, ax.get_ylim()[1] * 0.95,
            f'Mean: {diff_cond.mean():.3f}\nMedian: {diff_cond.median():.3f}',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))

    ax.set_xlabel('Comparison Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Distribution: Same vs Different Conditions',
                 fontsize=16)
    ax.grid(alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / 'same_vs_different_conditions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'same_vs_different_conditions.png'}")
    plt.close()

    # Print statistics
    print("\n--- Same vs Different Condition Statistics ---")
    print(f"Same condition: mean={same_cond.mean():.3f}, median={same_cond.median():.3f}, std={same_cond.std():.3f}")
    print(f"Different conditions: mean={diff_cond.mean():.3f}, median={diff_cond.median():.3f}, std={diff_cond.std():.3f}")
    print(f"Difference in means: {same_cond.mean() - diff_cond.mean():.3f}")

    # Effect size
    from scipy import stats
    cohens_d = (same_cond.mean() - diff_cond.mean()) / np.sqrt((same_cond.std()**2 + diff_cond.std()**2) / 2)
    print(f"Cohen's d: {cohens_d:.3f}")

    # Statistical test
    t_stat, p_val = stats.ttest_ind(same_cond, diff_cond)
    print(f"t-test: t={t_stat:.3f}, p={p_val:.2e}")


def plot_condition_specificity_matrix(subject_results, output_path):
    """
    Plot a matrix showing condition specificity for each subject.
    Specificity = (same-condition correlation) - (mean different-condition correlation)
    """
    all_conditions = list(subject_results[list(subject_results.keys())[0]]['correlation_matrix'].index)

    # Separate Shinobi and HCP
    shinobi_conds, hcp_conds = separate_shinobi_hcp_conditions(all_conditions)
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
    color_condition_labels(ax, ordered_conds, axis='y')

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
    """Generate all visualizations."""
    print("="*60)
    print("WITHIN-SUBJECT CONDITION CORRELATION VISUALIZATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data = load_data()

    subject_results = data['subject_results']
    avg_matrix = data['average']

    # Setup output directory
    output_path = Path(FIG_PATH) / 'within_subject_condition_correlations'
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_path}")

    # Generate plots
    print("\nGenerating visualizations...")

    # Individual subject heatmaps
    print("\n1. Individual subject heatmaps...")
    for subject in sorted(subject_results.keys()):
        plot_subject_heatmap(
            subject_results[subject]['correlation_matrix'],
            subject,
            output_path
        )

    # Average heatmap
    print("\n2. Average heatmap...")
    plot_average_heatmap(avg_matrix, output_path)

    # Same vs different comparison
    print("\n3. Same vs different condition comparison...")
    plot_same_vs_different_comparison(subject_results, output_path)

    # Condition specificity matrix
    print("\n4. Condition specificity matrix...")
    plot_condition_specificity_matrix(subject_results, output_path)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_path}")


if __name__ == '__main__':
    main()
