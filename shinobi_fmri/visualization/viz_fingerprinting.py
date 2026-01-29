"""
Fingerprinting analysis and visualization.

Computes fingerprinting scores (if needed) and creates plots showing:
1. Confusion matrix of nearest-neighbor subject identification
2. Within vs between subject correlation distributions
3. Fingerprinting scores by different groupings

Fingerprinting assesses whether brain maps are participant-specific by checking
if each map's most similar map (nearest neighbor) comes from the same subject.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys
from typing import Dict

# Add parent directory to path for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH, FIG_PATH
from shinobi_fmri.visualization.hcp_tasks import get_condition_label


def compute_fingerprinting_scores(corr_matrix: np.ndarray,
                                   subjects: list,
                                   sources: list,
                                   conditions: list) -> pd.DataFrame:
    """
    Compute fingerprinting scores.

    For each map:
    - Find the most similar map (highest correlation, excluding self)
    - Check if it comes from the same subject
    - Score = 1 if same subject, 0 otherwise

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix (n_maps x n_maps)
    subjects : list
        Subject ID for each map
    sources : list
        Source/level for each map (e.g., 'session-level', 'subject-level')
    conditions : list
        Condition for each map

    Returns
    -------
    pd.DataFrame
        Results with fingerprinting scores per map
    """
    n_maps = corr_matrix.shape[0]
    results = []

    print(f"Computing fingerprinting scores for {n_maps} maps...")

    for i in range(n_maps):
        # Get correlations for this map
        correlations = corr_matrix[i, :].copy()

        # Exclude self-correlation
        correlations[i] = -np.inf

        # Exclude NaN values
        valid_mask = ~np.isnan(correlations)

        if not np.any(valid_mask):
            # No valid correlations
            results.append({
                'map_idx': i,
                'subject': subjects[i],
                'source': sources[i],
                'condition': conditions[i],
                'nearest_neighbor_idx': np.nan,
                'nearest_neighbor_subject': np.nan,
                'nearest_neighbor_corr': np.nan,
                'is_same_subject': np.nan,
                'fingerprint_score': np.nan
            })
            continue

        # Find nearest neighbor (highest correlation)
        nn_idx = np.nanargmax(correlations)
        nn_corr = correlations[nn_idx]
        nn_subject = subjects[nn_idx]

        # Check if same subject
        is_same = (subjects[i] == nn_subject)

        results.append({
            'map_idx': i,
            'subject': subjects[i],
            'source': sources[i],
            'condition': conditions[i],
            'nearest_neighbor_idx': nn_idx,
            'nearest_neighbor_subject': nn_subject,
            'nearest_neighbor_corr': nn_corr,
            'is_same_subject': is_same,
            'fingerprint_score': 1.0 if is_same else 0.0
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_maps} maps...")

    return pd.DataFrame(results)


def aggregate_fingerprinting_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fingerprinting scores by different groupings.

    Parameters
    ----------
    df : pd.DataFrame
        Individual map results

    Returns
    -------
    pd.DataFrame
        Aggregated scores
    """
    aggregations = []

    # Overall score
    overall = df['fingerprint_score'].mean()
    aggregations.append({
        'grouping': 'overall',
        'group': 'all',
        'n_maps': len(df),
        'fingerprint_score': overall,
        'std': df['fingerprint_score'].std(),
        'sem': df['fingerprint_score'].sem()
    })

    # By subject
    for subj, subj_df in df.groupby('subject'):
        score = subj_df['fingerprint_score'].mean()
        aggregations.append({
            'grouping': 'by_subject',
            'group': subj,
            'n_maps': len(subj_df),
            'fingerprint_score': score,
            'std': subj_df['fingerprint_score'].std(),
            'sem': subj_df['fingerprint_score'].sem()
        })

    # By source (level)
    for source, source_df in df.groupby('source'):
        score = source_df['fingerprint_score'].mean()
        aggregations.append({
            'grouping': 'by_source',
            'group': source,
            'n_maps': len(source_df),
            'fingerprint_score': score,
            'std': source_df['fingerprint_score'].std(),
            'sem': source_df['fingerprint_score'].sem()
        })

    # By condition
    for cond, cond_df in df.groupby('condition'):
        score = cond_df['fingerprint_score'].mean()
        aggregations.append({
            'grouping': 'by_condition',
            'group': cond,
            'n_maps': len(cond_df),
            'fingerprint_score': score,
            'std': cond_df['fingerprint_score'].std(),
            'sem': cond_df['fingerprint_score'].sem()
        })

    # By subject and source
    for (subj, source), group_df in df.groupby(['subject', 'source']):
        score = group_df['fingerprint_score'].mean()
        aggregations.append({
            'grouping': 'by_subject_source',
            'group': f'{subj}_{source}',
            'n_maps': len(group_df),
            'fingerprint_score': score,
            'std': group_df['fingerprint_score'].std(),
            'sem': group_df['fingerprint_score'].sem()
        })

    return pd.DataFrame(aggregations)


def load_or_compute_fingerprinting_data():
    """
    Load fingerprinting results if they exist, otherwise compute them.

    Returns
    -------
    tuple
        (fp_detailed, fp_aggregated, corr_data)
    """
    processed_dir = Path(DATA_PATH) / 'processed'
    fp_dir = processed_dir / 'fingerprinting'
    detailed_file = fp_dir / 'fingerprinting_detailed.tsv'
    aggregated_file = fp_dir / 'fingerprinting_aggregated.tsv'

    # Load correlation matrix (always needed)
    corr_file = processed_dir / 'beta_maps_correlations.pkl'
    print(f"Loading correlation data from {corr_file}...")
    with open(corr_file, 'rb') as f:
        corr_data = pickle.load(f)

    print(f"Loaded {len(corr_data['mapnames'])} maps")

    # Check if fingerprinting results exist
    if detailed_file.exists() and aggregated_file.exists():
        print("Loading existing fingerprinting results...")
        fp_detailed = pd.read_csv(detailed_file, sep='\t')
        fp_aggregated = pd.read_csv(aggregated_file, sep='\t')
    else:
        print("\nFingerprinting results not found. Computing now...")
        print("="*60)

        # Create output directory
        fp_dir.mkdir(exist_ok=True, parents=True)

        # Extract metadata
        corr_matrix = corr_data['corr_matrix']
        subjects = corr_data['subj']
        sources = corr_data['source']
        conditions = corr_data['cond']

        # Compute fingerprinting scores
        fp_detailed = compute_fingerprinting_scores(
            corr_matrix, subjects, sources, conditions
        )

        # Save detailed results
        fp_detailed.to_csv(detailed_file, sep='\t', index=False)
        print(f"\nSaved detailed results to {detailed_file}")

        # Aggregate scores
        fp_aggregated = aggregate_fingerprinting_scores(fp_detailed)

        # Save aggregated results
        fp_aggregated.to_csv(aggregated_file, sep='\t', index=False)
        print(f"Saved aggregated results to {aggregated_file}")

        # Print summary
        print("\n" + "="*60)
        print("FINGERPRINTING ANALYSIS SUMMARY")
        print("="*60)

        overall_score = fp_aggregated[fp_aggregated['grouping'] == 'overall']['fingerprint_score'].values[0]
        print(f"\nOverall fingerprinting score: {overall_score:.3f}")
        print(f"(Proportion of maps where nearest neighbor is from same subject)")

        print("\n--- By Subject ---")
        subj_agg = fp_aggregated[fp_aggregated['grouping'] == 'by_subject'].sort_values('group')
        for _, row in subj_agg.iterrows():
            print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

        print("\n--- By Source/Level ---")
        source_agg = fp_aggregated[fp_aggregated['grouping'] == 'by_source'].sort_values('fingerprint_score', ascending=False)
        for _, row in source_agg.iterrows():
            print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

        print("\n--- By Condition ---")
        cond_agg = fp_aggregated[fp_aggregated['grouping'] == 'by_condition'].sort_values('fingerprint_score', ascending=False)
        for _, row in cond_agg.iterrows():
            print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

        print("\n" + "="*60)

    return fp_detailed, fp_aggregated, corr_data


def plot_confusion_matrix(fp_detailed, subjects, output_path):
    """
    Plot confusion matrix showing how often each subject's maps
    have nearest neighbors from each subject.
    """
    # Build confusion matrix
    confusion = pd.crosstab(
        fp_detailed['subject'],
        fp_detailed['nearest_neighbor_subject'],
        normalize='index'  # Row percentages
    )

    # Sort subjects
    subjects_sorted = sorted(subjects)
    confusion = confusion.reindex(index=subjects_sorted, columns=subjects_sorted, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        confusion,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Proportion of nearest neighbors'},
        ax=ax
    )

    ax.set_xlabel('Nearest Neighbor Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Map Subject', fontsize=12, fontweight='bold')
    ax.set_title('Subject Identification from Nearest Neighbors\n(Fingerprinting Confusion Matrix)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path / 'fingerprinting_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'fingerprinting_confusion_matrix.png'}")
    plt.close()


def plot_correlation_distributions(fp_detailed, corr_data, output_path):
    """
    Plot distributions of correlations within vs between subjects.
    """
    subjects = corr_data['subj']
    corr_matrix = corr_data['corr_matrix']

    # Get all pairwise correlations
    within_subject = []
    between_subject = []

    n_maps = len(subjects)
    for i in range(n_maps):
        for j in range(i + 1, n_maps):
            corr_val = corr_matrix[i, j]

            if np.isnan(corr_val):
                continue

            if subjects[i] == subjects[j]:
                within_subject.append(corr_val)
            else:
                between_subject.append(corr_val)

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(-0.5, 1.0, 50)

    ax.hist(between_subject, bins=bins, alpha=0.6, label='Between subjects',
            color='skyblue', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(within_subject, bins=bins, alpha=0.6, label='Within subject',
            color='salmon', density=True, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Within vs Between Subject Correlations',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    # Violin plot
    ax = axes[1]

    # Prepare data for violin plot
    data_for_violin = []
    labels_for_violin = []

    data_for_violin.extend(between_subject)
    labels_for_violin.extend(['Between\nSubjects'] * len(between_subject))

    data_for_violin.extend(within_subject)
    labels_for_violin.extend(['Within\nSubject'] * len(within_subject))

    violin_df = pd.DataFrame({
        'Correlation': data_for_violin,
        'Type': labels_for_violin
    })

    sns.violinplot(data=violin_df, x='Type', y='Correlation', ax=ax,
                   palette=['skyblue', 'salmon'], inner='box')

    # Add mean markers
    between_mean = np.mean(between_subject)
    within_mean = np.mean(within_subject)
    ax.plot(0, between_mean, 'D', color='darkblue', markersize=10,
            label=f'Mean = {between_mean:.3f}', zorder=10)
    ax.plot(1, within_mean, 'D', color='darkred', markersize=10,
            label=f'Mean = {within_mean:.3f}', zorder=10)

    ax.set_ylabel('Correlation (Pearson r)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Within vs Between Subject Similarity',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'fingerprinting_correlation_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'fingerprinting_correlation_distributions.png'}")
    plt.close()

    # Print statistics
    print("\n--- Correlation Statistics ---")
    print(f"Within subject:  mean={np.mean(within_subject):.3f}, std={np.std(within_subject):.3f}")
    print(f"Between subject: mean={np.mean(between_subject):.3f}, std={np.std(between_subject):.3f}")
    print(f"Effect size (Cohen's d): {(np.mean(within_subject) - np.mean(between_subject)) / np.sqrt((np.std(within_subject)**2 + np.std(between_subject)**2) / 2):.3f}")


def plot_fingerprinting_by_source(fp_aggregated, output_path):
    """Plot fingerprinting scores by source/level."""
    # Filter by source
    source_df = fp_aggregated[fp_aggregated['grouping'] == 'by_source'].copy()

    # Sort by score
    source_df = source_df.sort_values('fingerprint_score', ascending=True)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(source_df)))
    bars = ax.barh(source_df['group'], source_df['fingerprint_score'], color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (idx, row) in enumerate(source_df.iterrows()):
        ax.text(row['fingerprint_score'] + 0.01, i, f"{row['fingerprint_score']:.3f}  (n={int(row['n_maps'])})",
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Fingerprinting Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Data Source / Level', fontsize=12, fontweight='bold')
    ax.set_title('Fingerprinting Performance by Analysis Level',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect identification')
    ax.grid(alpha=0.3, linestyle='--', axis='x')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'fingerprinting_by_source.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'fingerprinting_by_source.png'}")
    plt.close()


def plot_fingerprinting_by_condition(fp_aggregated, output_path):
    """Plot fingerprinting scores by condition."""
    # Filter by condition
    cond_df = fp_aggregated[fp_aggregated['grouping'] == 'by_condition'].copy()

    # Separate Shinobi vs HCP conditions
    shinobi_conds = ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'UP', 'Kill', 'HealthLoss']
    cond_df['dataset'] = cond_df['group'].apply(lambda x: 'Shinobi' if x in shinobi_conds else 'HCP')

    # Sort by score within each dataset
    cond_df = cond_df.sort_values(['dataset', 'fingerprint_score'], ascending=[False, True])

    # Add icons to condition labels
    cond_df['group_with_icon'] = cond_df['group'].apply(get_condition_label)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by dataset
    colors = ['#FF6B6B' if ds == 'Shinobi' else '#4ECDC4' for ds in cond_df['dataset']]

    bars = ax.barh(range(len(cond_df)), cond_df['fingerprint_score'], color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (idx, row) in enumerate(cond_df.iterrows()):
        ax.text(row['fingerprint_score'] + 0.01, i, f"{row['fingerprint_score']:.3f}  (n={int(row['n_maps'])})",
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(cond_df)))
    ax.set_yticklabels(cond_df['group_with_icon'], fontsize=10)
    ax.set_xlabel('Fingerprinting Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    ax.set_title('Fingerprinting Performance by Condition',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect identification')
    ax.grid(alpha=0.3, linestyle='--', axis='x')

    # Add legend for datasets
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Shinobi'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='HCP'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Perfect identification')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'fingerprinting_by_condition.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'fingerprinting_by_condition.png'}")
    plt.close()


def plot_nearest_neighbor_correlations(fp_detailed, output_path):
    """Plot distribution of nearest neighbor correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of nearest neighbor correlations
    nn_corrs = fp_detailed['nearest_neighbor_corr'].dropna()

    ax.hist(nn_corrs, bins=50, color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.7)

    # Add mean line
    mean_corr = nn_corrs.mean()
    median_corr = nn_corrs.median()

    ax.axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_corr:.3f}')
    ax.axvline(median_corr, color='orange', linestyle='--', linewidth=2, label=f'Median = {median_corr:.3f}')

    ax.set_xlabel('Correlation with Nearest Neighbor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Nearest Neighbor Correlations\n(All maps to their most similar map)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / 'fingerprinting_nearest_neighbor_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'fingerprinting_nearest_neighbor_correlations.png'}")
    plt.close()

    # Print statistics
    print("\n--- Nearest Neighbor Correlation Statistics ---")
    print(f"Mean: {mean_corr:.3f}")
    print(f"Median: {median_corr:.3f}")
    print(f"Std: {nn_corrs.std():.3f}")
    print(f"Min: {nn_corrs.min():.3f}")
    print(f"Max: {nn_corrs.max():.3f}")


def main():
    """Generate all fingerprinting visualizations."""
    print("="*60)
    print("FINGERPRINTING ANALYSIS & VISUALIZATION")
    print("="*60)

    # Load or compute fingerprinting data
    print("\nPreparing data...")
    fp_detailed, fp_aggregated, corr_data = load_or_compute_fingerprinting_data()

    # Setup output directory
    output_path = Path(FIG_PATH) / 'fingerprinting'
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_path}")

    # Get unique subjects
    subjects = sorted(fp_detailed['subject'].unique())
    print(f"Subjects: {subjects}")

    # Generate plots
    print("\nGenerating visualizations...")

    print("\n1. Confusion matrix...")
    plot_confusion_matrix(fp_detailed, subjects, output_path)

    print("\n2. Correlation distributions...")
    plot_correlation_distributions(fp_detailed, corr_data, output_path)

    print("\n3. Fingerprinting by source/level...")
    plot_fingerprinting_by_source(fp_aggregated, output_path)

    print("\n4. Fingerprinting by condition...")
    plot_fingerprinting_by_condition(fp_aggregated, output_path)

    print("\n5. Nearest neighbor correlations...")
    plot_nearest_neighbor_correlations(fp_detailed, output_path)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_path}")


if __name__ == '__main__':
    main()
