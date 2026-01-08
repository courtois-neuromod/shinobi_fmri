#!/usr/bin/env python3
"""
Descriptive Statistics Visualization

Generates publication-ready 3-panel figure showing dataset descriptive statistics:
- Panel A: Events by subject and condition (grouped bar chart, spans top row)
- Panel B: Session/run availability matrix (heatmap, bottom left)
- Panel C: Volume counts distribution (box plot, bottom right)

Usage:
    python viz_descriptive_stats.py [-v] [--output path/to/figure.png] [--force]
    invoke descriptive.viz [--force]
"""
import os
import os.path as op
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import warnings

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.descriptive.dataset_summary import generate_dataset_summary
from shinobi_fmri.visualization.hcp_tasks import SHINOBI_COLOR

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def plot_events_subjects_x_conditions(ax, df, logger):
    """
    Plot events distribution per run (Panel A).

    Args:
        ax: Matplotlib axes object
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

    # Create Raincloud plot
    # x='subject' groups by subject
    # hue='Condition' creates clustered clouds within subject
    # dodge=True ensures they don't overlap each other
    # width_viol=0.4 makes clouds narrower to fit better
    # move=0.2 separates rain from cloud
    pt.RainCloud(x='subject', y='Count', hue='Condition', data=melted_df, ax=ax,
                 palette="Set2", 
                 width_viol=.4, width_box=0, box_showfliers=False,
                 alpha=.65, dodge=True, move=.2, point_size=2)

    # Styling
    ax.set_ylabel('Event Count per Run', fontsize=14)
    ax.set_xlabel('Subject', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Fix Legend: Move outside
    if ax.get_legend():
        ax.get_legend().remove()
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Place legend outside top right
    ax.legend(by_label.values(), by_label.keys(), 
              loc='upper left', bbox_to_anchor=(1, 1), title='Conditions',
              frameon=False, fontsize=10)

    # Panel label
    ax.text(-0.05, 1.05, 'A', transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')


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
    # Increase width ratio for marginal to fit legend outside if needed
    gs = gs_slot.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.1)
    ax_heatmap = fig.add_subplot(gs[0])
    ax_marginal = fig.add_subplot(gs[1]) # No sharey because Heatmap has margins now

    # --- Heatmap Data Preparation ---
    subjects = sorted(df['subject'].unique())
    sessions = sorted(df['session'].unique())

    # Pivot for Heatmap: Count unique runs
    # Add margins=True to get Total row/col
    runs_matrix = pd.pivot_table(df, values='run', index='subject', columns='session', 
                                 aggfunc='nunique', margins=True, margins_name='Total')
    
    # Reindex to ensure all sessions/subjects are present + Total
    # IMPORTANT: Include 'Total' in the reindex lists!
    idx_labels = subjects + ['Total']
    col_labels = sessions + ['Total']
    
    runs_matrix = runs_matrix.reindex(index=idx_labels, columns=col_labels).fillna(0).astype(int)

    # Plot Heatmap
    sns.heatmap(runs_matrix, ax=ax_heatmap, cmap='YlOrRd', annot=True, fmt='d',
                linewidths=0.5, linecolor='gray', cbar=False,
                square=False) # Auto-detect xticklabels/yticklabels

    ax_heatmap.set_xlabel('Session', fontsize=14)
    ax_heatmap.set_ylabel('Subject', fontsize=14)
    ax_heatmap.tick_params(axis='both', labelsize=10)
    
    # Panel B label
    ax_heatmap.text(-0.1, 1.05, 'B', transform=ax_heatmap.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
    
    # --- Marginal Plot Data Preparation ---
    # Group by Subject and Level
    level_stats = df.groupby(['subject', 'level']).agg({
        'cleared': 'sum',
        'run': 'count', 
        'duration': 'sum'
    }).reset_index()
    level_stats.rename(columns={'run': 'total_attempts'}, inplace=True)
    level_stats['failed'] = level_stats['total_attempts'] - level_stats['cleared']
    
    # Calculate Total for Dataset
    total_stats = df.groupby(['level']).agg({
        'cleared': 'sum',
        'run': 'count',
        'duration': 'sum'
    }).reset_index()
    total_stats['subject'] = 'Total'
    total_stats.rename(columns={'run': 'total_attempts'}, inplace=True)
    total_stats['failed'] = total_stats['total_attempts'] - total_stats['cleared']
    
    # Combine
    plot_data = pd.concat([level_stats, total_stats], ignore_index=True)
    
    level_colors = {
        'Level 1': '#1f77b4', 
        'Level 4': '#ff7f0e', 
        'Level 5': '#2ca02c'
    }
    
    plot_subjects = subjects + ['Total']
    y_positions = np.arange(len(plot_subjects)) + 0.5
    bar_height = 0.6
    
    for i, subj in enumerate(plot_subjects):
        subj_d = plot_data[plot_data['subject'] == subj]
        left_offset = 0
        
        for lvl in ['Level 1', 'Level 4', 'Level 5']:
            lvl_d = subj_d[subj_d['level'] == lvl]
            if len(lvl_d) > 0:
                cleared = lvl_d.iloc[0]['cleared']
                failed = lvl_d.iloc[0]['failed']
                base_color = level_colors.get(lvl, 'gray')
                
                # Cleared Segment
                if cleared > 0:
                    ax_marginal.barh(i + 0.5, cleared, height=bar_height, left=left_offset,
                                     color=base_color, alpha=1.0, edgecolor='white')
                    left_offset += cleared
                    
                # Failed Segment
                if failed > 0:
                    ax_marginal.barh(i + 0.5, failed, height=bar_height, left=left_offset,
                                     color=base_color, alpha=0.3, edgecolor='white', hatch='///')
                    left_offset += failed
        
        # Label
        total_n = subj_d['total_attempts'].sum()
        total_dur = subj_d['duration'].sum()
        dur_str = f"{int(total_dur//3600)}h"
        total_clr = subj_d['cleared'].sum()
        success_pct = (total_clr / total_n * 100) if total_n > 0 else 0
        
        label = f" {total_n} ({success_pct:.0f}%) | {dur_str}"
        ax_marginal.text(left_offset + 2, i + 0.5, label, va='center', fontsize=9, fontweight='bold' if subj == 'Total' else 'normal')

    # Styling Marginal
    # Align visually with heatmap rows. Heatmap has subjects + Total.
    # Heatmap 'Total' is at the bottom (index -1 in y-axis usually or index N).
    # Since we added margins=True, seaborn heatmap puts Total at the end.
    # So plot_subjects ordering (subjects + Total) matches heatmap y-axis order.
    
    ax_marginal.set_ylim(len(plot_subjects), 0)
    ax_marginal.set_yticks(y_positions)
    ax_marginal.set_yticklabels(plot_subjects)
    ax_marginal.yaxis.tick_right()
    ax_marginal.tick_params(axis='y', length=0)
    ax_marginal.spines['top'].set_visible(False)
    ax_marginal.spines['right'].set_visible(False)
    ax_marginal.spines['bottom'].set_visible(False)
    ax_marginal.spines['left'].set_visible(False)
    ax_marginal.set_xlabel('Number of Level Attempts', fontsize=12)
    
    # Legend outside
    legend_elements = []
    for lvl in ['Level 1', 'Level 4', 'Level 5']:
        legend_elements.append(plt.Rectangle((0,0),1,1, color=level_colors[lvl], label=lvl))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='k', label='Cleared'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, hatch='///', edgecolor='k', label='Failed'))
    
    ax_marginal.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9, title='Legend')

    # Panel C label
    ax_marginal.text(-0.1, 1.05, 'C', transform=ax_marginal.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')


def plot_volume_distribution(ax, df, logger):
    """
    Plot volume counts distribution by subject (Panel C).

    Args:
        ax: Matplotlib axes object
        df: Dataset summary DataFrame
        logger: AnalysisLogger instance
    """
    logger.debug("Generating Panel C: Volume counts distribution")

    # Prepare data for box plot
    subjects = sorted(df['subject'].unique())
    volume_data = []
    positions = []
    labels = []

    for i, subj in enumerate(subjects):
        subj_volumes = df[df['subject'] == subj]['n_volumes'].dropna().values
        if len(subj_volumes) > 0:
            volume_data.append(subj_volumes)
            positions.append(i)
            labels.append(subj)

    # Create box plot
    bp = ax.boxplot(volume_data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor=SHINOBI_COLOR, alpha=0.6, edgecolor='black', linewidth=1.2),
                     whiskerprops=dict(color='black', linewidth=1.2),
                     capprops=dict(color='black', linewidth=1.2))

    # Add overall mean line
    overall_mean = df['n_volumes'].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall Mean: {overall_mean:.1f}', alpha=0.7)

    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Number of Volumes', fontsize=14)
    ax.set_xlabel('Subject', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel label
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')


def create_figure(csv_path, output_path, logger):
    """
    Create 2-panel figure with descriptive statistics.

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

    # Create figure with GridSpec: 2 rows, 1 column
    # We use a wider figure to accommodate the marginal plot in Panel B
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.3)

    # Panel A: Top
    ax_events = fig.add_subplot(gs[0])
    
    # Panel B: Bottom (Composite Heatmap + Marginal)
    # Passed as a GridSpec slot to the plotting function
    
    # Generate panels
    plot_events_subjects_x_conditions(ax_events, df_unique_runs, logger)
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

    # Create figure (force regeneration if --force is used)
    if args.force or not op.exists(output_path):
        create_figure(csv_path, output_path, logger)
    else:
        logger.info(f"Figure already exists: {output_path}. Use --force to regenerate.")
        print(f"Figure already exists: {output_path}")
        print(f"Use --force to regenerate")

    logger.close()

    # Print summary if figure was generated
    if args.force or not op.exists(output_path):
        print(f"✓ Generated descriptive statistics figure")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
