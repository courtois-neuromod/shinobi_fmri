"""
Within-subject condition correlation analysis.

For each subject, compute how maps from different conditions correlate with each other.
This shows the specificity of different experimental conditions within individuals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

# Add parent directory to path for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH


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


def main():
    """Main analysis."""

    # Paths
    processed_dir = Path(DATA_PATH) / 'processed'
    correlation_file = processed_dir / 'beta_maps_correlations.pkl'
    output_dir = processed_dir / 'within_subject_condition_correlations'
    output_dir.mkdir(exist_ok=True)

    # Load data
    corr_data = load_correlation_data(correlation_file)

    # Compute within-subject condition correlations
    subject_results, unique_conditions = compute_within_subject_condition_correlations(corr_data)

    # Save individual subject results
    for subject, data in subject_results.items():
        output_file = output_dir / f'{subject}_condition_correlations.tsv'
        data['correlation_matrix'].to_csv(output_file, sep='\t')
        print(f"\nSaved {subject} results to {output_file}")

    # Aggregate across subjects
    avg_df = aggregate_across_subjects(subject_results, unique_conditions)

    # Save aggregated results
    avg_file = output_dir / 'average_condition_correlations.tsv'
    avg_df.to_csv(avg_file, sep='\t')
    print(f"\nSaved average results to {avg_file}")

    # Save full results as pickle
    pickle_file = output_dir / 'within_subject_condition_correlations.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'subject_results': subject_results,
            'average': avg_df,
            'unique_conditions': unique_conditions
        }, f)
    print(f"Saved full results to {pickle_file}")

    return subject_results, avg_df


if __name__ == '__main__':
    subject_results, avg_df = main()
