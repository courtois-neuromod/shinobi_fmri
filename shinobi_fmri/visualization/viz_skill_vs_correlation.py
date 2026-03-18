"""
Visualize game skill metrics against average within-subject beta map
correlations.

For each subject, computes the mean same-condition intra-subject Pearson
correlation from the pre-computed beta correlation matrix, then plots it
against each of the three skill metrics (clear rate, average progression,
efficiency).
"""

import argparse
import json
import os
import os.path as op
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH, FIG_PATH, SUBJECTS


# Conditions to exclude (not scientifically meaningful)
EXCLUDED_CONDITIONS = ["UP"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_skill_metrics(skill_path: str) -> Dict[str, Dict[str, Any]]:
    """Load skill metrics JSON produced by ``compute_skill_metrics.py``.

    Args:
        skill_path: Path to ``skill_metrics.json``.

    Returns:
        ``{subject: {clear_rate, avg_progression, avg_efficiency, ...}}``.
    """
    with open(skill_path) as f:
        return json.load(f)


def load_beta_correlations(corr_path: str) -> Dict[str, Any]:
    """Load the beta correlation pickle.

    Args:
        corr_path: Path to ``beta_maps_correlations.pkl``.

    Returns:
        Dictionary with keys ``corr_matrix``, ``subj``, ``cond``, etc.
    """
    with open(corr_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Within-subject correlation computation
# ---------------------------------------------------------------------------

def compute_avg_within_subject_correlation(
    corr_data: Dict[str, Any],
    subjects: List[str],
) -> Dict[str, float]:
    """Compute average same-condition intra-subject correlation per subject.

    For each subject and each condition, collects pairwise correlations
    between all session-level beta maps of that condition, then averages
    across conditions to produce a single reliability number.

    Only session-level maps are used (partial/Xrun maps excluded).

    Args:
        corr_data: Loaded beta correlation data.
        subjects: Subject IDs to process.

    Returns:
        ``{subject: mean_pearson_r}``.
    """
    corr_matrix = corr_data["corr_matrix"]
    all_subjs = np.array(corr_data["subj"])
    all_conds = np.array(corr_data["cond"])
    file_list = corr_data.get("mapnames", corr_data["fnames"])

    # Exclude partial-session maps (e.g. "2runs.nii.gz")
    valid_indices = [
        i
        for i in range(len(file_list))
        if not re.search(r"\d+run\.nii\.gz", file_list[i])
    ]

    unique_conds = sorted(
        set(all_conds[i] for i in valid_indices)
        - set(EXCLUDED_CONDITIONS)
    )

    results: Dict[str, float] = {}

    for subj in subjects:
        cond_means: List[float] = []
        for cond in unique_conds:
            indices = [
                idx
                for idx in valid_indices
                if all_subjs[idx] == subj and all_conds[idx] == cond
            ]
            if len(indices) < 2:
                continue

            pairs: List[float] = []
            for i_pos in range(len(indices)):
                for j_pos in range(i_pos + 1, len(indices)):
                    r = corr_matrix[indices[i_pos], indices[j_pos]]
                    if not np.isnan(r):
                        pairs.append(r)

            if pairs:
                cond_means.append(float(np.mean(pairs)))

        results[subj] = float(np.mean(cond_means)) if cond_means else np.nan

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "clear_rate": "Clear rate",
    "avg_progression": "Average progression (%)",
    "avg_efficiency": "Efficiency (progression / damage)",
}


def plot_skill_vs_correlation(
    skill_data: Dict[str, Dict[str, Any]],
    within_corr: Dict[str, float],
    subjects: List[str],
    output_path: str,
) -> plt.Figure:
    """Create a 1x3 scatter plot of skill metrics vs. within-subject correlation.

    Args:
        skill_data: Skill metrics per subject.
        within_corr: Average within-subject correlation per subject.
        subjects: Ordered subject list.
        output_path: Where to save the figure.

    Returns:
        The matplotlib Figure.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    palette = sns.color_palette("Set2", n_colors=len(subjects))
    metric_keys = ["clear_rate", "avg_progression", "avg_efficiency"]

    for ax, metric_key in zip(axes, metric_keys):
        x_vals = [skill_data[sub][metric_key] for sub in subjects]
        y_vals = [within_corr[sub] for sub in subjects]

        for i, sub in enumerate(subjects):
            ax.scatter(
                x_vals[i],
                y_vals[i],
                color=palette[i],
                s=120,
                zorder=3,
                edgecolors="white",
                linewidth=1.2,
            )
            ax.annotate(
                sub,
                (x_vals[i], y_vals[i]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
                color=palette[i],
            )

        ax.set_xlabel(METRIC_LABELS[metric_key], fontsize=10)
        ax.set_ylabel("Mean within-subject correlation (r)", fontsize=10)

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Game skill vs. within-subject beta-map reliability",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    output_dir = op.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot game skill metrics vs. within-subject beta-map correlation."
    )
    parser.add_argument(
        "--skill-input",
        default=None,
        help="Path to skill_metrics.json (default: auto from config)",
    )
    parser.add_argument(
        "--corr-input",
        default=None,
        help="Path to beta_maps_correlations.pkl (default: auto from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: {FIG_PATH}/skill_vs_correlation.png)",
    )
    args = parser.parse_args()

    skill_path = args.skill_input or op.join(
        DATA_PATH, "processed", "skill_metrics", "skill_metrics.json"
    )
    corr_path = args.corr_input or op.join(
        DATA_PATH, "processed", "beta_maps_correlations.pkl"
    )
    output_path = args.output or op.join(FIG_PATH, "skill_vs_correlation.png")

    if not op.exists(skill_path):
        print(
            f"Error: Skill metrics not found at {skill_path}\n"
            "Run  invoke behav.skill-metrics  first."
        )
        sys.exit(1)
    if not op.exists(corr_path):
        print(f"Error: Correlation data not found at {corr_path}")
        sys.exit(1)

    print("Loading skill metrics...")
    skill_data = load_skill_metrics(skill_path)

    print("Loading beta correlations...")
    corr_data = load_beta_correlations(corr_path)

    print("Computing average within-subject correlations...")
    within_corr = compute_avg_within_subject_correlation(corr_data, SUBJECTS)
    for sub in SUBJECTS:
        print(f"  {sub}: r = {within_corr[sub]:.4f}")

    print("Generating figure...")
    plot_skill_vs_correlation(skill_data, within_corr, SUBJECTS, output_path)


if __name__ == "__main__":
    main()
