
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle
import numpy as np
import os
import argparse
import re
from matplotlib.collections import LineCollection
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.hcp_tasks import (
    TASK_ICONS, TASK_COLORS, SHINOBI_COLOR,
    get_event_to_task_mapping, get_task_label
)

def process_beta_correlations_data(pickle_path):
    """
    Loads beta correlations data from a pickle file and processes it into DataFrames.

    Parameters:
    - pickle_path: str, path to the .pkl file containing the data dictionary.

    Returns:
    - plot_df: pandas DataFrame for main analysis (Intra/Inter correlations).
    - consistency_df: pandas DataFrame for consistency analysis (Intra-subject reliability of Partial maps).
    """
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        datadict = pickle.load(f)

    # Ensure we use mapnames if available, otherwise fnames
    file_list = datadict.get('mapnames', datadict['fnames'])
    
    n_runs = len(file_list)
    conds = datadict['cond']
    subjs = datadict['subj']
    sess = datadict['ses'] 

    # --- Part 1: Main Analysis (Intra/Inter) - Excluding Xrun maps ---
    
    valid_indices = []
    for i in range(n_runs):
        fname = file_list[i]
        if not re.search(r'\d+run\.nii\.gz', fname):
            valid_indices.append(i)
    
    print(f"Total maps: {n_runs}. Maps for main analysis (excluding Xrun): {len(valid_indices)}")

    corr_r = []
    corr_cond = []
    corr_intera = []

    unique_conds = np.unique(conds)
    unique_subjs = np.unique(subjs)

    print("Processing intra-subject correlations...")
    for cond in unique_conds:
        for subj in unique_subjs:
            relevant_indices = [idx for idx in valid_indices if conds[idx] == cond and subjs[idx] == subj]
            for i_idx, i in enumerate(relevant_indices):
                for j in relevant_indices[i_idx+1:]: 
                    corr_r.append(datadict['corr_matrix'][i, j])
                    corr_cond.append(cond)
                    corr_intera.append('intra-subject')

    print("Processing inter-subject correlations...")
    for cond in unique_conds:
        for subj in unique_subjs:
            idxs_i = [idx for idx in valid_indices if conds[idx] == cond and subjs[idx] == subj]
            idxs_j = [idx for idx in valid_indices if conds[idx] == cond and subjs[idx] != subj]
            for i in idxs_i:
                for j in idxs_j:
                    if i < j:
                        corr_r.append(datadict['corr_matrix'][i, j])
                        corr_cond.append(cond)
                        corr_intera.append('inter-subject')

    print("Processing Inter-annotations...")
    # Intra-subject (diff conds)
    for cond in unique_conds:
        for subj in unique_subjs:
            idxs_i = [idx for idx in valid_indices if conds[idx] == cond and subjs[idx] == subj]
            idxs_j = [idx for idx in valid_indices if conds[idx] != cond and subjs[idx] == subj]
            for i in idxs_i:
                for j in idxs_j:
                    if i < j:
                        corr_r.append(datadict['corr_matrix'][i, j])
                        corr_cond.append('Inter')
                        corr_intera.append('intra-subject')

    # Inter-subject (diff conds)
    for cond in unique_conds:
        for subj in unique_subjs:
            idxs_i = [idx for idx in valid_indices if conds[idx] == cond and subjs[idx] == subj]
            idxs_j = [idx for idx in valid_indices if conds[idx] != cond and subjs[idx] != subj]
            for i in idxs_i:
                for j in idxs_j:
                    if i < j:
                        corr_r.append(datadict['corr_matrix'][i, j])
                        corr_cond.append('Inter')
                        corr_intera.append('inter-subject')

    plot_df = pd.DataFrame({'r': corr_r, 'event': corr_cond, 'comparison': corr_intera})

    # --- Part 2: Consistency Analysis (Intra-subject reliability of Xrun maps) ---
    print("Processing Consistency Analysis (Intra-subject reliability)...")

    # Extract 'source' field if available (newer format)
    sources = datadict.get('source', [None] * n_runs)

    map_meta = []

    for i in range(n_runs):
        source = sources[i]
        n_run = None

        # Try to extract run count from 'source' field (e.g., 'session-level_2runs')
        if source and '_' in source:
            match = re.search(r'_(\d+)runs?', source)
            if match:
                n_run = int(match.group(1))

        # Fallback: try to extract from filename (legacy format)
        if n_run is None:
            fname = file_list[i]
            match = re.search(r'_(\d+)runs?\.nii\.gz', fname)
            if match:
                n_run = int(match.group(1))

        # Only include maps with a valid run count (partial session maps)
        if n_run is not None:
            map_meta.append({
                'idx': i,
                'subj': subjs[i],
                'ses': sess[i],
                'cond': conds[i],
                'runs': n_run
            })
            
    meta_df = pd.DataFrame(map_meta)
    
    cons_subjs = []
    cons_conds = []
    cons_runs = []
    cons_r = []
    
    if not meta_df.empty:
        # Group by Subject, Condition, Runs
        groups = meta_df.groupby(['subj', 'cond', 'runs'])
        
        for (subj, cond, n_run), group in groups:
            indices = group['idx'].values
            if len(indices) < 2:
                continue 
            
            # Calculate pairwise correlations between all sessions for this specific condition & run count
            for i_local in range(len(indices)):
                for j_local in range(i_local + 1, len(indices)):
                    idx_a = indices[i_local]
                    idx_b = indices[j_local]
                    
                    r_val = datadict['corr_matrix'][idx_a, idx_b]
                    
                    cons_subjs.append(subj)
                    cons_conds.append(cond)
                    cons_runs.append(n_run)
                    cons_r.append(r_val)

    consistency_df = pd.DataFrame({
        'subject': cons_subjs,
        'condition': cons_conds,
        'num_runs': cons_runs,
        'r': cons_r
    })
    
    return plot_df, consistency_df

def plot_beta_correlations(plot_df, consistency_df, output_path=None):
    """
    Generates the combined figure.
    """
    plot_df = plot_df.copy()
    plot_df['r'] = pd.to_numeric(plot_df['r'], errors='coerce')
    plot_df = plot_df.dropna(subset=['r'])

    # --- Setup Colors and Logic ---
    highlight_events = ['Kill', 'HIT', 'JUMP', 'HealthLoss', 'DOWN', 'RIGHT', 'LEFT', 'Inter']

    # Use centralized HCP task configuration
    highlight_color = SHINOBI_COLOR
    colors_dict = TASK_COLORS
    task_icons = TASK_ICONS
    event_to_task = get_event_to_task_mapping()

    plot_df = plot_df[~plot_df['event'].str.contains('-') | (plot_df['event'] == 'Inter')]

    def create_palette(events):
        return {event: highlight_color if event in highlight_events else '#808080' for event in events}

    inter_df = plot_df[plot_df['comparison'] == 'inter-subject']
    intra_df = plot_df[plot_df['comparison'] == 'intra-subject']

    def get_sorted_events(df):
        if df.empty: return []
        event_means = df.groupby('event')['r'].mean().reset_index()
        return event_means.sort_values('r', ascending=False)['event'].tolist()

    inter_events = get_sorted_events(inter_df)
    intra_events = get_sorted_events(intra_df)
    inter_palette = create_palette(inter_events)
    intra_palette = create_palette(intra_events)

    # --- Plotting Layout ---
    sns.set_style("white")
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Wider figure to accommodate legends on the right
    fig = plt.figure(figsize=(10.08, 12), dpi=300)
    # Compact height ratios for A and B, adjusted spacing
    gs = fig.add_gridspec(3, 5, height_ratios=[0.6, 0.6, 0.7], width_ratios=[1, 1, 1, 1, 0.3], hspace=0.9, wspace=0.4)

    # Plots A and B only span first 4 columns, leaving space for legends
    ax_intra = fig.add_subplot(gs[0, :4])
    ax_inter = fig.add_subplot(gs[1, :4])
    
    # --- Row 1: Intra-subject ---
    sns.pointplot(data=intra_df, x='event', y='r', palette=intra_palette, ax=ax_intra, order=intra_events, ci='sd', capsize=0.1, join=True)
    ax_intra.set_title('Within-participant Correlations') # Unbold
    ax_intra.set_xlabel('')
    ax_intra.set_ylabel('Mean Pearson r')
    ax_intra.set_ylim(-0.2, 1.05)

    def set_xtick_labels(ax, events):
        """Set tick labels with icons and colors."""
        # Get current labels and extract event names
        labels = ax.get_xticklabels()
        event_names = [label.get_text() for label in labels]

        new_labels = []
        for event in event_names:
            if event in highlight_events:
                # Shinobi events - no icon
                new_labels.append(event)
            else:
                task = event_to_task.get(event)
                if task and task in task_icons:
                    # HCP events - add icon
                    icon = task_icons[task]
                    new_labels.append(f'{icon} {event}')
                else:
                    new_labels.append(event)

        # Set new labels
        ax.set_xticklabels(new_labels, rotation=90)

        # Color the labels based on original event names
        for label, event in zip(ax.get_xticklabels(), event_names):
            if event in highlight_events:
                label.set_color(highlight_color)
                label.set_fontweight('bold')
            else:
                task = event_to_task.get(event)
                if task and task in colors_dict:
                    label.set_color(colors_dict[task])
                else:
                    label.set_color('black')

    set_xtick_labels(ax_intra, intra_events)
    ax_intra.grid(axis='y', linestyle='-', alpha=0.7)
    for spine in ax_intra.spines.values():
        spine.set_visible(False)
    
    # --- Row 2: Inter-subject ---
    sns.pointplot(data=inter_df, x='event', y='r', palette=inter_palette, ax=ax_inter, order=inter_events, ci='sd', capsize=0.1, join=True)
    ax_inter.set_title('Between-participant Correlations') # Unbold
    ax_inter.set_xlabel('')
    ax_inter.set_ylabel('Mean Pearson r')
    ax_inter.set_ylim(-0.2, 1.05)
    set_xtick_labels(ax_inter, inter_events)
    ax_inter.grid(False)
    for y_val in [0, 0.25, 0.5, 0.75, 1.0]:
        ax_inter.axhline(y=y_val, color='gray', linestyle='-', alpha=0.3, zorder=0)
        ax_intra.axhline(y=y_val, color='gray', linestyle='-', alpha=0.3, zorder=0)
    for spine in ax_inter.spines.values():
        spine.set_visible(False)

    # --- Row 3: Consistency (Intra-subject Reliability) ---
    subjects = sorted(consistency_df['subject'].unique())
    labels_bottom = ['C', 'D', 'E', 'F']
    events_ordered = ['Kill', 'HIT', 'JUMP', 'HealthLoss', 'DOWN', 'RIGHT', 'LEFT']
    
    max_runs_global = int(consistency_df['num_runs'].max()) if not consistency_df.empty else 1
    run_palette = sns.color_palette("Oranges", n_colors=max_runs_global + 3)[3:]

    for idx, subj in enumerate(subjects):
        if idx >= 4: break
        # Bottom plots also span only first 4 columns
        ax_sub = fig.add_subplot(gs[2, idx if idx < 4 else idx])
        
        subj_df = consistency_df[consistency_df['subject'] == subj]
        agg_df = subj_df.groupby(['condition', 'num_runs'])['r'].agg(['mean', 'std']).reset_index()
        
        x_centers = []
        x_labels_plot = []
        current_x_start = 0
        run_x_scale = 0.3 
        event_spacing = 1.5
        
        for evt in events_ordered:
            evt_data = agg_df[agg_df['condition'] == evt].sort_values('num_runs')
            if evt_data.empty: continue
            
            runs = evt_data['num_runs'].values
            means = evt_data['mean'].values
            
            xs = current_x_start + (runs - runs[0]) * run_x_scale
            
            points = np.array([xs, means]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            seg_colors = []
            for i in range(len(runs) - 1):
                r_idx_mid = (runs[i] + runs[i+1]) / 2
                c_idx = min(int(r_idx_mid) - 1, len(run_palette) - 1)
                c_idx = max(0, c_idx)
                seg_colors.append(run_palette[c_idx])
            
            lc = LineCollection(segments, colors=seg_colors, linewidth=2, alpha=0.8, zorder=1)
            ax_sub.add_collection(lc)
            
            for x, y, r_n in zip(xs, means, runs):
                c_idx = min(int(r_n) - 1, len(run_palette) - 1)
                c_idx = max(0, c_idx)
                color = run_palette[c_idx]
                ax_sub.scatter(x, y, color=color, s=30, zorder=3, edgecolors='white', linewidth=0.5)

            if len(xs) > 0:
                x_centers.append(np.mean(xs))
                x_labels_plot.append(evt)
                current_x_start = xs[-1] + event_spacing
        
        ax_sub.set_title(f'{subj}') # Unbold
        ax_sub.set_xticks(x_centers)
        ax_sub.set_xticklabels(x_labels_plot, rotation=90, fontsize=8)
        
        if idx == 0:
            ax_sub.set_ylabel('Intra-subject Pearson r')
        else:
            ax_sub.set_ylabel('')
            
        ax_sub.set_ylim(0, 1.05)
        ax_sub.grid(axis='y', linestyle='--', alpha=0.5)
        ax_sub.set_xlabel('')
        
        # Remove frame (spines)
        for spine in ax_sub.spines.values():
            spine.set_visible(False)
        
        label_char = labels_bottom[idx] if idx < len(labels_bottom) else ''
        ax_sub.text(-0.15, 1.15, label_char, transform=ax_sub.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

    # --- Labels A, B ---
    ax_intra.text(-0.05, 1.15, 'A', transform=ax_intra.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    ax_inter.text(-0.05, 1.15, 'B', transform=ax_inter.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Legend text (positioned so the middle of the list aligns with middle of A and B)
    # List has 8 items total (Tasks header + Shinobi + 6 tasks), spanning 7 intervals of 0.03 = 0.21
    # Middle is at 3.5 intervals = 0.105 from top
    # Middle of A and B graphs is approximately at y=0.58
    legend_middle = 0.58
    legend_top = legend_middle + 0.105

    fig.text(0.82, legend_top, 'Tasks:', fontsize=10, fontweight='bold', ha='left')
    fig.text(0.82, legend_top - 0.03, 'Shinobi', fontsize=10, fontweight='bold', color=highlight_color, ha='left')
    y_pos = legend_top - 0.06
    for task, color in colors_dict.items():
        task_label = get_task_label(task)
        fig.text(0.82, y_pos, task_label, fontsize=10, color=color, ha='left')
        y_pos -= 0.03

    # New Legend for Number of Runs (positioned at the level of C-F row)
    run_legend_elements = []
    runs_to_show = min(max_runs_global, len(run_palette))
    for i in range(runs_to_show):
        color = run_palette[i]
        run_legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{i+1}',
                                          markerfacecolor=color, markersize=8))

    fig.legend(handles=run_legend_elements, title='Number of runs',
               loc='upper left', bbox_to_anchor=(0.82, 0.25), frameon=False,
               title_fontsize=10, fontsize=10)

    # Tight layout with more room on the right for legends
    plt.tight_layout(rect=[0, 0, 0.80, 1]) 
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate beta correlations figure.")
    parser.add_argument("--input",
                       help="Path to the input pickle file (default: derived from config)")
    parser.add_argument("--output",
                       help="Path to save the output figure (default: derived from config)")
    parser.add_argument("--low-level", action="store_true",
                       help="Use correlation matrix from processed_low-level")

    args = parser.parse_args()

    # Import config for default paths
    try:
        from shinobi_fmri.config import DATA_PATH, FIG_PATH
        
        if args.low_level:
            default_input = os.path.join(DATA_PATH, "processed_low-level", "beta_maps_correlations.pkl")
            # Adjust output path based on input type
            fig_path_adjusted = FIG_PATH.replace('figures', 'figures_raw_low-level')
            default_output = os.path.join(fig_path_adjusted, "beta_correlations_plot_low-level.png")
        else:
            default_input = os.path.join(DATA_PATH, "processed", "beta_maps_correlations.pkl")
            # Adjust output path based on input type
            fig_path_adjusted = FIG_PATH.replace('figures', 'figures_raw')
            default_output = os.path.join(fig_path_adjusted, "beta_correlations_plot.png")
            
    except ImportError:
        default_input = None
        default_output = None

    input_path = args.input if args.input else default_input
    output_path = args.output if args.output else default_output

    if input_path is None:
        print("Error: Input path not specified and config not available.")
        exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        exit(1)

    df_main, df_consistency = process_beta_correlations_data(input_path)
    plot_beta_correlations(df_main, df_consistency, output_path=output_path)
