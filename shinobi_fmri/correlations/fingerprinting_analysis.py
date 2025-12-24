"""
Fingerprinting analysis for brain maps.

Assesses whether brain maps are participant-specific by checking if each map's
most similar map (nearest neighbor) comes from the same subject.

Fingerprinting score = proportion of maps where nearest neighbor is from same subject
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple
import sys

# Add parent directory to path for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH


def load_correlation_data(correlation_file: Path) -> Dict:
    """Load the correlation matrix and metadata."""
    print(f"Loading correlation data from {correlation_file}...")
    with open(correlation_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['mapnames'])} maps")
    print(f"Correlation matrix shape: {data['corr_matrix'].shape}")

    return data


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

    print(f"\nComputing fingerprinting scores for {n_maps} maps...")

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


def main():
    """Main fingerprinting analysis."""

    # Paths
    processed_dir = Path(DATA_PATH) / 'processed'
    correlation_file = processed_dir / 'beta_maps_correlations.pkl'
    output_dir = processed_dir / 'fingerprinting'
    output_dir.mkdir(exist_ok=True)

    # Load correlation data
    data = load_correlation_data(correlation_file)

    # Extract metadata
    corr_matrix = data['corr_matrix']
    subjects = data['subj']
    sources = data['source']
    conditions = data['cond']

    # Compute fingerprinting scores
    results_df = compute_fingerprinting_scores(
        corr_matrix, subjects, sources, conditions
    )

    # Save detailed results
    detailed_file = output_dir / 'fingerprinting_detailed.tsv'
    results_df.to_csv(detailed_file, sep='\t', index=False)
    print(f"\nSaved detailed results to {detailed_file}")

    # Aggregate scores
    agg_df = aggregate_fingerprinting_scores(results_df)

    # Save aggregated results
    agg_file = output_dir / 'fingerprinting_aggregated.tsv'
    agg_df.to_csv(agg_file, sep='\t', index=False)
    print(f"Saved aggregated results to {agg_file}")

    # Print summary
    print("\n" + "="*60)
    print("FINGERPRINTING ANALYSIS SUMMARY")
    print("="*60)

    overall_score = agg_df[agg_df['grouping'] == 'overall']['fingerprint_score'].values[0]
    print(f"\nOverall fingerprinting score: {overall_score:.3f}")
    print(f"(Proportion of maps where nearest neighbor is from same subject)")

    print("\n--- By Subject ---")
    subj_agg = agg_df[agg_df['grouping'] == 'by_subject'].sort_values('group')
    for _, row in subj_agg.iterrows():
        print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

    print("\n--- By Source/Level ---")
    source_agg = agg_df[agg_df['grouping'] == 'by_source'].sort_values('fingerprint_score', ascending=False)
    for _, row in source_agg.iterrows():
        print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

    print("\n--- By Condition ---")
    cond_agg = agg_df[agg_df['grouping'] == 'by_condition'].sort_values('fingerprint_score', ascending=False)
    for _, row in cond_agg.iterrows():
        print(f"  {row['group']}: {row['fingerprint_score']:.3f} (n={row['n_maps']})")

    print("\n" + "="*60)

    return results_df, agg_df


if __name__ == '__main__':
    results_df, agg_df = main()
