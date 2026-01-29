#!/usr/bin/env python3
"""
MVPA Confusion Matrix Visualization.

Creates a 2x2 grid of confusion matrices for all subjects with task icons
and proper task grouping/separation.
"""
import os
import os.path as op
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import warnings

# Suppress warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")

import shinobi_fmri.config as config
from shinobi_fmri.visualization.hcp_tasks import (
    TASK_COLORS, SHINOBI_COLOR, TASK_ICONS, SHINOBI_CONDITIONS,
    get_event_to_task_mapping, get_condition_label, get_task_label
)


def load_decoder_data(subject, mvpa_results_path):
    """
    Load decoder data for a given subject.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    mvpa_results_path : str
        Path to MVPA results directory

    Returns
    -------
    decoder : object
        Fitted decoder with classes_ attribute
    fold_confusions : np.ndarray
        Confusion matrices for each CV fold
    """
    decoder_fname = f'{subject}_decoder.pkl'
    decoder_fpath = op.join(mvpa_results_path, decoder_fname)
    print(f'Loading decoder data from {subject}')

    with open(decoder_fpath, 'rb') as f:
        file = pickle.load(f)

    return file['decoder'], file['confusion_matrices']['fold_confusions']


def plot_confusion_matrix(ax, subject, mvpa_results_path, show_colorbar=False, cbar_ax=None):
    """
    Plot confusion matrix for a single subject.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    subject : str
        Subject ID
    mvpa_results_path : str
        Path to MVPA results
    show_colorbar : bool
        Whether to show colorbar
    cbar_ax : matplotlib.axes.Axes
        Axes for colorbar
    """
    decoder, fold_confusions = load_decoder_data(subject, mvpa_results_path)

    # Get classes and event-to-task mapping
    classes = decoder.classes_
    event_to_task = get_event_to_task_mapping()

    # Average confusion matrix across folds and normalize
    avg_cm = np.mean(fold_confusions, axis=0)
    normalized_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

    # Reorder classes: Shinobi first, then grouped by task
    shinobi_classes = [c for c in classes if c in SHINOBI_CONDITIONS]
    task_classes = []
    for task_name in TASK_COLORS.keys():
        task_events = [c for c in classes
                      if event_to_task.get(c) == task_name
                      and c not in SHINOBI_CONDITIONS]
        task_classes.extend(task_events)

    reordered_classes = shinobi_classes + task_classes
    reorder_indices = [list(classes).index(c) for c in reordered_classes]
    normalized_cm = normalized_cm[np.ix_(reorder_indices, reorder_indices)]

    # Create formatted labels with icons for HCP conditions
    tick_labels = []
    for condition in reordered_classes:
        if condition in SHINOBI_CONDITIONS:
            tick_labels.append(condition)
        else:
            # Add task icon
            task_name = event_to_task.get(condition)
            if task_name and task_name in TASK_ICONS:
                icon = TASK_ICONS[task_name]
                tick_labels.append(f'{icon} {condition}')
            else:
                tick_labels.append(condition)

    # Plot heatmap
    heatmap = sns.heatmap(
        normalized_cm,
        annot=False,
        cmap='YlGnBu',
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar=show_colorbar,
        cbar_ax=cbar_ax
    )

    # Add "Accuracy" label to colorbar if showing
    if show_colorbar and cbar_ax is not None:
        cbar_ax.set_ylabel('Accuracy', rotation=270, labelpad=20, fontsize=14)
        # Increase colorbar tick label size
        cbar_ax.tick_params(labelsize=12)

    # Add task separator lines
    n_shinobi = len(shinobi_classes)

    # Draw Shinobi box
    ax.plot([0, n_shinobi], [0, 0], color=SHINOBI_COLOR, linewidth=3, clip_on=False)  # top
    ax.plot([0, n_shinobi], [n_shinobi, n_shinobi], color=SHINOBI_COLOR, linewidth=3, clip_on=False)  # bottom
    ax.plot([0, 0], [0, n_shinobi], color=SHINOBI_COLOR, linewidth=3, clip_on=False)  # left
    ax.plot([n_shinobi, n_shinobi], [0, n_shinobi], color=SHINOBI_COLOR, linewidth=3, clip_on=False)  # right

    # Add task block separators for HCP tasks
    prev_task = None
    for idx, condition in enumerate(reordered_classes[n_shinobi:], start=n_shinobi):
        current_task = event_to_task.get(condition)
        if current_task != prev_task and prev_task is not None:
            # Draw separator line at the boundary where new task starts
            ax.axhline(y=idx, color='gray', linewidth=1.5, alpha=0.5)
            ax.axvline(x=idx, color='gray', linewidth=1.5, alpha=0.5)
        if current_task != prev_task:
            prev_task = current_task

    # Color tick labels
    for label in ax.get_xticklabels():
        text = label.get_text()
        # Extract condition name (remove icon if present)
        condition = text.split(' ', 1)[-1] if ' ' in text else text

        if condition in SHINOBI_CONDITIONS:
            label.set_color(SHINOBI_COLOR)
            label.set_fontweight('bold')
        else:
            task_name = event_to_task.get(condition)
            if task_name:
                label.set_color(TASK_COLORS[task_name])
            else:
                label.set_color('#333333')
        label.set_size(12)
        label.set_rotation(90)

    for label in ax.get_yticklabels():
        text = label.get_text()
        # Extract condition name (remove icon if present)
        condition = text.split(' ', 1)[-1] if ' ' in text else text

        if condition in SHINOBI_CONDITIONS:
            label.set_color(SHINOBI_COLOR)
            label.set_fontweight('bold')
        else:
            task_name = event_to_task.get(condition)
            if task_name:
                label.set_color(TASK_COLORS[task_name])
            else:
                label.set_color('#333333')
        label.set_size(12)

    ax.set_title(f"{subject}", fontsize=18, y=1.05)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)


def create_confusion_matrix_figure(subjects, mvpa_results_path, output_path=None):
    """
    Create 2x2 grid of confusion matrices.

    Parameters
    ----------
    subjects : list
        List of subject IDs
    mvpa_results_path : str
        Path to MVPA results directory
    output_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # Add colorbar axis
    cbar_ax = fig.add_axes([.92, .3, .02, .4])

    # Plot each subject
    for i, subject in enumerate(subjects):
        show_cbar = (i == 0)
        plot_confusion_matrix(
            axes[i], subject, mvpa_results_path,
            show_colorbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None
        )
        # Remove redundant labels
        if i in [0, 1]:
            axes[i].set_xlabel('')
        if i in [1, 3]:
            axes[i].set_ylabel('')

    # Create legend with task colors
    legend_elements = []
    for task_name, color in TASK_COLORS.items():
        task_label = get_task_label(task_name)
        legend_elements.append(Line2D([0], [0], color='w', label=task_label, markerfacecolor='w', markersize=0))

    # Add Shinobi to legend
    legend_elements.append(Line2D([0], [0], color='w', label='Shinobi', markerfacecolor='w', markersize=0))

    # Create legend
    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=14
    )

    # Color legend text
    for i, text in enumerate(legend.get_texts()):
        if i < len(TASK_COLORS):
            task_name = list(TASK_COLORS.keys())[i]
            text.set_color(TASK_COLORS[task_name])
        else:
            text.set_color(SHINOBI_COLOR)
            text.set_fontweight('bold')

    # Add "Tasks" title
    fig.text(0.5, 0.05, 'Tasks', fontweight='bold', ha='center', fontsize=16)

    # Layout adjustment
    plt.tight_layout(rect=[0, 0.1, 0.9, 1])

    if output_path:
        os.makedirs(op.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MVPA confusion matrices")
    parser.add_argument("--screening", type=int, default=20,
                        help="Screening percentile used (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure (default: auto-generated)")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Do not show the figure (default: False)")
    parser.add_argument("--no-low-level", action="store_true",
                        help="Exclude low-level features from visualization (default: False, low-level features are included)")

    args = parser.parse_args()

    # Setup paths
    path_to_data = config.DATA_PATH
    # Always use processed directory (low-level features are now default)
    processed_dir = "processed"
    mvpa_results_path = op.join(path_to_data, processed_dir, f"mvpa_results_s{args.screening}")
    subjects = config.SUBJECTS

    # Default output path
    if args.output is None:
        # Use standard figures directory
        figures_path = config.FIG_PATH
        if 'figures' in figures_path:
            output_dir = figures_path
        else:
            output_dir = op.join(os.getcwd(), "reports", "figures")

        os.makedirs(output_dir, exist_ok=True)
        # Include map type (raw/corrected) in filename
        args.output = op.join(output_dir, f"mvpa_confusion_matrices_s{args.screening}_raw.png")

    print(f"Creating confusion matrix figure...")
    print(f"Results path: {mvpa_results_path}")
    print(f"Subjects: {subjects}")

    fig = create_confusion_matrix_figure(subjects, mvpa_results_path, args.output)
    
    if not args.no_show:
        plt.show()
    
    plt.close(fig)
