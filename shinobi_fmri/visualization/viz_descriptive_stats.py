#!/usr/bin/env python3
"""
Descriptive Statistics Visualization

Generates publication-ready 3-panel figure showing dataset descriptive statistics:
- Panel A: Events by subject and condition (boxplot)
- Panel B: Session/run availability matrix (heatmap)
- Panel C: Level attempts by subject (stacked bar)

Usage:
    python viz_descriptive_stats.py [-v] [--output path/to/figure.png] [--force]
    invoke viz.descriptive [--force]
"""
import os
import os.path as op
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.descriptive.dataset_summary import generate_dataset_summary
from shinobi_fmri.visualization.hcp_tasks import SHINOBI_COLOR

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def plot_events_subjects_x_conditions(fig, gs_slot, df, logger):
    """
    Plot events distribution per run (Panel A) - One boxplot per subject.

    Args:
        fig: Matplotlib Figure object
        gs_slot: GridSpec slot for this panel
        df: Dataset summary DataFrame
        logger: AnalysisLogger instance
    """
    logger.debug("Generating Panel A: Events distribution per run")

    # Melt dataframe to long format
    event_cols = [c for c in df.columns if c.startswith('n_') and c not in ['n_volumes']]
    melted_df = df.melt(id_vars=['subject', 'session', 'run'], 
                        value_vars=event_cols,
                        var_name='Condition', 
                        value_name='Count')
    
    # Clean condition names
    melted_df['Condition'] = melted_df['Condition'].str.replace('n_', '')
    
    # Get subjects and conditions
    subjects = sorted(melted_df['subject'].unique())
    conditions = sorted(melted_df['Condition'].unique())
    n_subjects = len(subjects)
    
    # Create subplots for each subject with shared Y axis
    gs_inner = gs_slot.subgridspec(1, n_subjects, wspace=0.05)
    axes = [fig.add_subplot(gs_inner[i]) for i in range(n_subjects)]
    
    # Get global y limits for shared axis
    y_max = melted_df['Count'].max() * 1.1
    
    for i, (ax, subj) in enumerate(zip(axes, subjects)):
        subj_data = melted_df[melted_df['subject'] == subj]
        
        # Create boxplot - all orange
        bp = ax.boxplot([subj_data[subj_data['Condition'] == cond]['Count'].values 
                         for cond in conditions],
                        positions=range(len(conditions)),
                        widths=0.6,
                        patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=SHINOBI_COLOR, alpha=0.5),
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor=SHINOBI_COLOR, alpha=0.7, edgecolor='black', linewidth=1),
                        whiskerprops=dict(color='black', linewidth=1),
                        capprops=dict(color='black', linewidth=1))
        
        # X-axis: condition names in bold orange
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=8, fontweight='bold', color=SHINOBI_COLOR, rotation=45, ha='right')
        
        # Subject title
        subj_short = subj.replace('sub-', '')
        ax.set_title(subj_short, fontsize=10, fontweight='bold', color=SHINOBI_COLOR)
        
        # Shared Y axis
        ax.set_ylim(0, y_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if i == 0:
            # First plot: show Y axis labels
            ax.set_ylabel('Event count per run', fontsize=10)
        else:
            # Other plots: hide Y tick labels but keep axis
            ax.tick_params(axis='y', labelleft=False)
            ax.spines['left'].set_visible(False)
    
    # Panel label on first axis
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes,
                 fontsize=14, fontweight='bold', va='top', ha='right')


def plot_composite_session_stats(fig, gs_slot, df, logger):
    """
    Plot composite session statistics (Panel B & C): Heatmap + Marginal Summary.

    Args:
        fig: Matplotlib Figure object
        gs_slot: GridSpec slot for bottom row
        df: Dataset summary DataFrame (long format, per level attempt)
        logger: AnalysisLogger instance
    """
    logger.debug("Generating Panel B/C: Composite session statistics")

    # Create nested GridSpec: Heatmap (left) and Marginal (right)
    gs = gs_slot.subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    ax_heatmap = fig.add_subplot(gs[0])
    ax_marginal = fig.add_subplot(gs[1])

    # --- Heatmap Data Preparation ---
    subjects = sorted(df['subject'].unique())
    sessions = sorted(df['session'].unique())

    # Pivot for Heatmap: Count unique runs per subject/session
    runs_matrix = pd.pivot_table(df, values='run', index='subject', columns='session', 
                                 aggfunc='nunique').fillna(0).astype(int)
    
    # Add Total column (sum of runs per subject)
    runs_matrix['Total'] = runs_matrix.sum(axis=1)
    
    # Add Total row (sum of runs per session)
    total_row = runs_matrix.sum(axis=0)
    runs_matrix.loc['Total'] = total_row
    
    # Simplify session labels: ses-001 -> 001
    new_cols = [c.replace('ses-', '') if c != 'Total' else c for c in runs_matrix.columns]
    runs_matrix.columns = new_cols
    
    # Simplify subject labels: sub-01 -> 01
    new_idx = [i.replace('sub-', '') if i != 'Total' else i for i in runs_matrix.index]
    runs_matrix.index = new_idx

    # Plot Heatmap
    sns.heatmap(runs_matrix, ax=ax_heatmap, cmap='YlOrRd', annot=True, fmt='d',
                linewidths=0.5, linecolor='gray', cbar=False,
                annot_kws={'size': 8})

    ax_heatmap.set_xlabel('Session', fontsize=11, fontweight='bold', color=SHINOBI_COLOR)
    ax_heatmap.set_ylabel('Subject', fontsize=11, fontweight='bold', color=SHINOBI_COLOR)
    ax_heatmap.tick_params(axis='both', labelsize=8)
    # Rotate x labels for better fit
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    
    # Panel B label
    ax_heatmap.text(-0.12, 1.02, 'B', transform=ax_heatmap.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # --- Marginal Plot (Panel C): Level attempts by subject ---
    # Group by Subject and Level, count unique runs
    level_stats = df.groupby(['subject', 'level']).agg({
        'cleared': 'sum',
        'run': 'nunique',  # Count unique runs, not rows
        'duration': 'sum'
    }).reset_index()
    level_stats.rename(columns={'run': 'n_runs'}, inplace=True)
    
    # Calculate Total for Dataset
    total_stats = df.groupby(['level']).agg({
        'cleared': 'sum',
        'run': 'nunique',
        'duration': 'sum'
    }).reset_index()
    total_stats['subject'] = 'Total'
    total_stats.rename(columns={'run': 'n_runs'}, inplace=True)
    
    # Combine
    plot_data = pd.concat([level_stats, total_stats], ignore_index=True)
    
    level_colors = {
        'Level 1': '#1f77b4', 
        'Level 4': '#ff7f0e', 
        'Level 5': '#2ca02c'
    }
    
    # Simplify subject labels
    plot_subjects = [s.replace('sub-', '') for s in subjects] + ['Total']
    y_positions = np.arange(len(plot_subjects))
    bar_height = 0.6
    
    for i, subj in enumerate(plot_subjects):
        subj_key = f'sub-{subj}' if subj != 'Total' else 'Total'
        subj_d = plot_data[plot_data['subject'] == subj_key]
        left_offset = 0
        
        for lvl in ['Level 1', 'Level 4', 'Level 5']:
            lvl_d = subj_d[subj_d['level'] == lvl]
            if len(lvl_d) > 0:
                n_runs = lvl_d.iloc[0]['n_runs']
                cleared = lvl_d.iloc[0]['cleared']
                base_color = level_colors.get(lvl, 'gray')
                
                if n_runs > 0:
                    ax_marginal.barh(i, n_runs, height=bar_height, left=left_offset,
                                     color=base_color, alpha=0.85, edgecolor='white', linewidth=0.5)
                    left_offset += n_runs
        
        # Compact label: just total runs
        total_runs = subj_d['n_runs'].sum()
        total_dur = subj_d['duration'].sum()
        dur_str = f"{int(total_dur//3600)}h" if total_dur > 0 else ""
        
        label = f" {int(total_runs)}" + (f" ({dur_str})" if dur_str else "")
        ax_marginal.text(left_offset + 1, i, label, va='center', fontsize=8, 
                        fontweight='bold' if subj == 'Total' else 'normal')

    # Styling Marginal
    ax_marginal.set_ylim(len(plot_subjects) - 0.5, -0.5)
    ax_marginal.set_yticks(y_positions)
    ax_marginal.set_yticklabels(plot_subjects, fontsize=8)
    ax_marginal.set_xlabel('Number of Runs', fontsize=11, fontweight='bold', color=SHINOBI_COLOR)
    ax_marginal.tick_params(axis='y', length=0)
    ax_marginal.spines['top'].set_visible(False)
    ax_marginal.spines['right'].set_visible(False)
    ax_marginal.spines['left'].set_visible(False)
    ax_marginal.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Legend - positioned carefully to avoid overlap
    legend_elements = [plt.Rectangle((0,0),1,1, color=level_colors[lvl], label=lvl) 
                       for lvl in ['Level 1', 'Level 4', 'Level 5']]
    ax_marginal.legend(handles=legend_elements, loc='lower right', fontsize=8, 
                       frameon=True, framealpha=0.9, edgecolor='none')

    # Panel C label
    ax_marginal.text(-0.15, 1.02, 'C', transform=ax_marginal.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')


def create_figure(csv_path, output_path, logger):
    """
    Create compact 3-panel figure with descriptive statistics.

    Args:
        csv_path: Path to dataset summary CSV
        output_path: Path to save output figure
        logger: AnalysisLogger instance
    """
    logger.info("Creating descriptive statistics figure")

    # Load data (per level attempt)
    if not op.exists(csv_path):
        raise FileNotFoundError(f"Dataset summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset summary with {len(df)} level attempts")
    
    # Deduplicate for Panel A (which expects per-run data, not per-level)
    df_unique_runs = df.drop_duplicates(subset=['subject', 'session', 'run'])

    # Create compact figure with GridSpec: 2 rows, 1 column
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.4, height_ratios=[1, 1])

    # Generate panels - both use GridSpec slots now
    plot_events_subjects_x_conditions(fig, gs[0], df_unique_runs, logger)
    plot_composite_session_stats(fig, gs[1], df, logger)

    # Save figure
    os.makedirs(op.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics visualization figure"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path from config (default: use config.DATA_PATH)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: {FIG_PATH}/descriptive_stats.png)"
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to dataset_summary.csv (default: auto-detect or generate)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of both CSV and figure"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (0=WARNING, 1=INFO, 2=DEBUG). Can be repeated: -v, -vv"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Custom directory for log files (default: ./logs/)"
    )

    args = parser.parse_args()

    # Setup paths
    data_path = args.data_path if args.data_path else config.DATA_PATH
    output_path = args.output if args.output else op.join(config.FIG_PATH, "descriptive_stats.png")
    csv_path = args.csv_path if args.csv_path else op.join(
        data_path, "processed", "descriptive", "dataset_summary.csv"
    )

    # Setup logger
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = AnalysisLogger(
        log_name="viz_descriptive_stats",
        verbosity=verbosity_map.get(args.verbose, logging.DEBUG),
        log_dir=args.log_dir
    )

    # Generate CSV if needed
    if args.force or not op.exists(csv_path):
        if args.force:
            logger.info("Force regeneration: generating dataset summary CSV")
        else:
            logger.info("Dataset summary CSV not found, generating it")

        generate_dataset_summary(
            data_path=data_path,
            output_csv=csv_path,
            logger=logger
        )

    # Create figure
    create_figure(csv_path, output_path, logger)

    # Close logger without printing summary (not useful for single-figure generation)
    for handler in logger.logger.handlers:
        handler.close()
        logger.logger.removeHandler(handler)

    print(f"✓ Generated descriptive statistics figure")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
