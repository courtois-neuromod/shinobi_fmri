#!/usr/bin/env python3
"""
Descriptive Annotations Visualization

Generates a figure comparing event counts and durations across subjects and annotations:
- Panel A: Count per session, showing variance across sessions (all conditions)
- Panel B: Event durations, showing variance across occurrences (excludes Kill/HealthLoss)

Usage:
    python viz_descriptive_annotations.py [-v] [--output-dir path/to/folder]
    invoke viz.descriptive-annotations
"""
import os
import os.path as op
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from glob import glob

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.visualization.hcp_tasks import SHINOBI_COLOR

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Conditions to analyze (game annotations)
# Order: RIGHT, LEFT, DOWN, HIT, JUMP, then outcomes (Kill, HealthLoss)
CONDITIONS = ['RIGHT', 'LEFT', 'DOWN', 'HIT', 'JUMP', 'Kill', 'HealthLoss']
CONDITIONS_DURATION = ['RIGHT', 'LEFT', 'DOWN', 'HIT', 'JUMP']  # Exclude Kill/HealthLoss


def load_all_events(data_path):
    """
    Load all annotated events from the dataset.
    
    Args:
        data_path: Root data directory path
        
    Returns:
        events_df: DataFrame with all events (subject, session, run, trial_type, onset, duration)
    """
    pattern = op.join(data_path, "shinobi", "sub-*", "ses-*", "func", "*_desc-annotated_events.tsv")
    event_files = glob(pattern)
    
    all_events = []
    for fpath in event_files:
        # Parse subject/session/run from filename
        basename = op.basename(fpath)
        parts = basename.split('_')
        sub = parts[0]
        ses = parts[1]
        run = parts[3].replace('run-', '')
        
        # Load events
        try:
            df = pd.read_csv(fpath, sep='\t', low_memory=False)
            if 'trial_type' in df.columns and 'duration' in df.columns:
                df = df[['trial_type', 'onset', 'duration']].copy()
                df['subject'] = sub
                df['session'] = ses
                df['run'] = run
                all_events.append(df)
        except Exception as e:
            continue
    
    if not all_events:
        raise ValueError("No valid event files found")
    
    events_df = pd.concat(all_events, ignore_index=True)
    
    # Filter to conditions of interest
    events_df = events_df[events_df['trial_type'].isin(CONDITIONS)]
    
    return events_df


def prepare_count_data(events_df):
    """
    Prepare count data: number of events per session, for variance across sessions.
    
    Returns:
        count_df: DataFrame with subject, session, condition, count
    """
    # Count events per subject/session/condition
    count_df = events_df.groupby(['subject', 'session', 'trial_type']).size().reset_index(name='count')
    count_df.columns = ['subject', 'session', 'condition', 'count']
    count_df['subject_short'] = count_df['subject'].str.replace('sub-', '')
    
    return count_df


def prepare_duration_data(events_df):
    """
    Prepare duration data: individual event durations for variance across occurrences.
    
    Returns:
        duration_df: DataFrame with subject, condition, duration (one row per event)
    """
    duration_df = events_df[['subject', 'trial_type', 'duration']].copy()
    duration_df.columns = ['subject', 'condition', 'duration']
    duration_df['subject_short'] = duration_df['subject'].str.replace('sub-', '')
    
    return duration_df


def create_annotations_figure(count_df, duration_df, output_path):
    """
    Create the main annotations figure with boxplots.
    Panel A: Count per session (boxplot across sessions) - all conditions
    Panel B: Duration per event (boxplot across occurrences) - excludes Kill/HealthLoss
    Compact version for quarter A4 page.
    """
    # Quarter A4: ~105mm x 74mm = ~4.1" x 2.9"
    fig, axes = plt.subplots(1, 2, figsize=(4.1, 2.5))
    
    subjects = sorted(count_df['subject_short'].unique())
    # Use ordered conditions from global
    conditions_count = [c for c in CONDITIONS if c in count_df['condition'].unique()]
    conditions_duration = [c for c in CONDITIONS_DURATION if c in duration_df['condition'].unique()]
    
    # Filter dataframes to only include ordered conditions
    count_plot_df = count_df[count_df['condition'].isin(conditions_count)].copy()
    count_plot_df['condition'] = pd.Categorical(count_plot_df['condition'], 
                                                 categories=conditions_count, ordered=True)
    
    duration_plot_df = duration_df[duration_df['condition'].isin(conditions_duration)].copy()
    duration_plot_df['condition'] = pd.Categorical(duration_plot_df['condition'], 
                                                    categories=conditions_duration, ordered=True)
    
    # Panel A: Count boxplots (all conditions) using seaborn
    ax = axes[0]
    sns.boxplot(x='condition', y='count', hue='subject_short', data=count_plot_df,
                ax=ax, palette='Set2', order=conditions_count, hue_order=subjects,
                linewidth=0.3, fliersize=0, width=0.7)
    
    ax.set_xlabel('')
    ax.set_ylabel('Count', fontsize=5)
    ax.set_title('Count per session', fontsize=6)
    ax.legend_.remove()
    ax.set_xticklabels(conditions_count, fontsize=5, fontweight='bold', color=SHINOBI_COLOR, 
                       rotation=90, ha='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.3)
    ax.tick_params(axis='both', labelsize=5, width=0.3, length=2)
    # Add panel label A
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    
    # Panel B: Duration boxplots (exclude Kill/HealthLoss) using seaborn
    ax = axes[1]
    sns.boxplot(x='condition', y='duration', hue='subject_short', data=duration_plot_df,
                ax=ax, palette='Set2', order=conditions_duration, hue_order=subjects,
                linewidth=0.3, fliersize=0, width=0.7)
    
    ax.set_xlabel('')
    ax.set_ylabel('Seconds', fontsize=5)
    ax.set_title('Press durations', fontsize=6)
    ax.set_ylim(0, 6)  # Max 6 seconds
    ax.set_xticklabels(conditions_duration, fontsize=5, fontweight='bold', color=SHINOBI_COLOR, 
                       rotation=90, ha='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.3)
    ax.tick_params(axis='both', labelsize=5, width=0.3, length=2)
    # Add panel label B
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    
    # Legend - compact, from the right panel
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), 
              title='Subject', frameon=False, fontsize=4, title_fontsize=5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_annotations_figure(data_path, output_dir, logger):
    """
    Generate the annotations count and duration figure.
    
    Args:
        data_path: Root data directory
        output_dir: Directory to save figures
        logger: AnalysisLogger instance
    """
    logger.info("Loading event data from annotated events files")
    
    # Load all events directly from files
    events_df = load_all_events(data_path)
    logger.info(f"Loaded {len(events_df)} events from {events_df['subject'].nunique()} subjects")
    
    # Prepare data
    count_df = prepare_count_data(events_df)
    duration_df = prepare_duration_data(events_df)
    
    logger.info(f"Count data: {len(count_df)} subject×session×condition combinations")
    logger.info(f"Duration data: {len(duration_df)} individual events")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the figure
    print("Generating annotations figure...")
    
    create_annotations_figure(count_df, duration_df, op.join(output_dir, "annotations_count_duration.png"))
    
    logger.info(f"Figure saved to: {output_dir}")
    print(f"\n✓ Generated annotations figure in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotations count and duration figure"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path from config"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {FIG_PATH}/descriptive_annotations)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Custom log directory"
    )

    args = parser.parse_args()

    # Setup paths
    data_path = args.data_path if args.data_path else config.DATA_PATH
    output_dir = args.output_dir if args.output_dir else op.join(config.FIG_PATH, "descriptive_annotations")

    # Setup logger
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = AnalysisLogger(
        log_name="viz_descriptive_annotations",
        verbosity=verbosity_map.get(args.verbose, logging.DEBUG),
        log_dir=args.log_dir
    )

    # Generate the annotations figure
    generate_annotations_figure(data_path, output_dir, logger)

    # Close logger without summary
    for handler in logger.logger.handlers:
        handler.close()
        logger.logger.removeHandler(handler)


if __name__ == "__main__":
    main()
