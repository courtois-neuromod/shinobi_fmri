#!/usr/bin/env python3
"""
Aggregate permutation results and compute p-values for MVPA.
"""
import os
import os.path as op
import pickle
import numpy as np
import argparse
import shinobi_fmri.config as config


def aggregate_permutation_results(subject, mvpa_results_path, n_permutations, logger=None):
    """
    Aggregate all permutation results for a subject and compute p-values.

    Args:
        subject: Subject ID (e.g., 'sub-01')
        mvpa_results_path: Path to MVPA results directory
        n_permutations: Expected number of permutations
        logger: Optional logger

    Returns:
        dict with aggregated results and p-values
    """
    perm_dir = op.join(mvpa_results_path, f"{subject}_permutations")

    if not op.exists(perm_dir):
        raise FileNotFoundError(f"Permutation directory not found: {perm_dir}")

    # Load actual results
    decoder_path = op.join(mvpa_results_path, f"{subject}_decoder.pkl")
    if not op.exists(decoder_path):
        raise FileNotFoundError(f"Decoder results not found: {decoder_path}")

    with open(decoder_path, 'rb') as f:
        actual_results = pickle.load(f)

    actual_scores_per_class = actual_results['scores_per_class']
    class_labels = list(actual_scores_per_class.keys())

    # Initialize storage for permuted scores
    permuted_scores_per_class = {label: [] for label in class_labels}

    # Collect all permutation files
    perm_files = [f for f in os.listdir(perm_dir) if f.startswith('perm_') and f.endswith('.pkl')]

    if not perm_files:
        raise FileNotFoundError(f"No permutation result files found in {perm_dir}")

    print(f"Found {len(perm_files)} permutation result files")

    # Load all permutation results
    total_perms_loaded = 0
    for perm_file in sorted(perm_files):
        perm_path = op.join(perm_dir, perm_file)
        with open(perm_path, 'rb') as f:
            perm_results_batch = pickle.load(f)

        for perm_result in perm_results_batch:
            perm_scores = perm_result['scores_per_class']

            for class_label in class_labels:
                if class_label in perm_scores:
                    # Take mean across CV folds for this permutation
                    mean_score = np.mean(perm_scores[class_label])
                    permuted_scores_per_class[class_label].append(mean_score)

            total_perms_loaded += 1

    print(f"Loaded {total_perms_loaded} permutations")

    if total_perms_loaded < n_permutations:
        print(f"WARNING: Expected {n_permutations} permutations, but only found {total_perms_loaded}")

    # Compute p-values for each class
    p_values = {}
    for class_label in class_labels:
        actual_mean = np.mean(actual_scores_per_class[class_label])
        permuted_scores = np.array(permuted_scores_per_class[class_label])

        # P-value: proportion of permuted scores >= actual score
        p_value = np.mean(permuted_scores >= actual_mean)
        p_values[class_label] = p_value

        print(f"{class_label}: actual={actual_mean:.3f}, p={p_value:.4f}")

    # Compute overall p-value (using mean across classes)
    overall_actual = np.mean([np.mean(scores) for scores in actual_scores_per_class.values()])
    overall_permuted = [np.mean([permuted_scores_per_class[c][i] for c in class_labels])
                        for i in range(total_perms_loaded)]
    overall_p_value = np.mean(np.array(overall_permuted) >= overall_actual)

    print(f"\nOverall: actual={overall_actual:.3f}, p={overall_p_value:.4f}")

    # Create aggregated results
    aggregated = {
        'subject': subject,
        'n_permutations': total_perms_loaded,
        'class_labels': class_labels,
        'actual_scores_per_class': actual_scores_per_class,
        'permuted_scores_per_class': permuted_scores_per_class,
        'p_values_per_class': p_values,
        'overall_actual_score': overall_actual,
        'overall_p_value': overall_p_value
    }

    # Save aggregated results
    output_path = op.join(mvpa_results_path, f"{subject}_permutation_pvalues.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(aggregated, f)

    print(f"\nSaved aggregated results to: {output_path}")

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate MVPA permutation results")
    parser.add_argument("-s", "--subject", required=True,
                        help="Subject ID (e.g., sub-01)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Expected number of permutations (default: 1000)")
    parser.add_argument("--screening", type=int, default=20,
                        help="Screening percentile used (default: 20)")

    args = parser.parse_args()

    path_to_data = config.DATA_PATH
    mvpa_results_path = op.join(path_to_data, "processed", f"mvpa_results_s{args.screening}")

    print(f"Aggregating permutation results for {args.subject}")
    print(f"Results path: {mvpa_results_path}")
    print(f"Expected permutations: {args.n_permutations}\n")

    aggregate_permutation_results(args.subject, mvpa_results_path, args.n_permutations)
