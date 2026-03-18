"""
Visualize per-session game skill against per-session brain-map reliability.

For each (subject, session, condition) triplet, computes:
- **Skill**: mean composite score from ``compute_session_skill.py``
- **Reliability**: mean Pearson correlation of that session's beta map
  with all other sessions of the same subject and condition

Produces a multi-panel scatter plot (one panel per condition), hued by
subject, with per-subject regression trend lines.
"""

import argparse
import json
import math
import os
import os.path as op
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import (
    CONDITIONS,
    DATA_PATH,
    FIG_PATH,
    LOW_LEVEL_CONDITIONS,
    SUBJECTS,
)


# Conditions to exclude from plots (not scientifically meaningful)
EXCLUDED_CONDITIONS = ["UP"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_skill(skill_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load session-level skill metrics JSON.

    Args:
        skill_path: Path to ``session_skill_metrics.json``.

    Returns:
        ``{subject: {session: {mean_composite, n_reps, ...}}}``.
    """
    with open(skill_path) as f:
        return json.load(f)


def load_beta_correlations(corr_path: str) -> Dict[str, Any]:
    """Load the beta correlation pickle.

    Args:
        corr_path: Path to ``beta_maps_correlations.pkl``.

    Returns:
        Dictionary with keys ``corr_matrix``, ``subj``, ``ses``,
        ``cond``, ``source``, etc.
    """
    with open(corr_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Per-session brain reliability
# ---------------------------------------------------------------------------

def compute_per_session_reliability(
    corr_data: Dict[str, Any],
    subjects: List[str],
) -> pd.DataFrame:
    """Compute per-session reliability from beta correlation matrix.

    For each (subject, session, condition) where ``source == "session-level"``,
    computes the mean correlation with all *other* sessions of the same
    subject and condition.

    Args:
        corr_data: Loaded beta correlation data.
        subjects: Subject IDs to include.

    Returns:
        DataFrame with columns ``subject``, ``session``, ``condition``,
        ``reliability``.
    """
    corr_matrix = corr_data["corr_matrix"]
    all_subjs = np.array(corr_data["subj"])
    all_sessions = np.array(corr_data["ses"])
    all_conds = np.array(corr_data["cond"])
    all_sources = np.array(corr_data["source"])
    file_list = corr_data.get("mapnames", corr_data["fnames"])

    # Keep only session-level maps, excluding partial-run maps
    valid_mask = np.array([
        s == "session-level"
        and not re.search(r"\d+run\.nii\.gz", file_list[i])
        for i, s in enumerate(all_sources)
    ])

    valid_indices = np.where(valid_mask)[0]

    # Build lookup: (subject, condition) -> list of (session, index)
    lookup: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for idx in valid_indices:
        sub = all_subjs[idx]
        cond = all_conds[idx]
        ses = all_sessions[idx]
        if sub not in subjects:
            continue
        if cond in EXCLUDED_CONDITIONS:
            continue
        lookup.setdefault((sub, cond), []).append((ses, idx))

    rows: List[Dict[str, Any]] = []
    for (sub, cond), entries in lookup.items():
        if len(entries) < 2:
            continue
        for i, (ses_i, idx_i) in enumerate(entries):
            other_corrs = []
            for j, (ses_j, idx_j) in enumerate(entries):
                if i == j:
                    continue
                r = corr_matrix[idx_i, idx_j]
                if not np.isnan(r):
                    other_corrs.append(r)
            if other_corrs:
                rows.append({
                    "subject": sub,
                    "session": ses_i,
                    "condition": cond,
                    "reliability": float(np.mean(other_corrs)),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def build_merged_dataframe(
    skill_data: Dict[str, Dict[str, Dict[str, Any]]],
    reliability_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join session skill and session reliability on (subject, session).

    Skill is condition-agnostic (one value per session), while reliability
    is condition-specific.  The merged frame replicates skill across
    conditions.

    Args:
        skill_data: ``{subject: {session: {mean_composite, ...}}}``.
        reliability_df: Output of ``compute_per_session_reliability``.

    Returns:
        DataFrame with columns ``subject``, ``session``, ``condition``,
        ``skill``, ``reliability``.
    """
    skill_rows = []
    for sub in skill_data:
        for ses in skill_data[sub]:
            skill_rows.append({
                "subject": sub,
                "session": ses,
                "skill": skill_data[sub][ses]["mean_composite"],
            })
    skill_df = pd.DataFrame(skill_rows)

    merged = reliability_df.merge(skill_df, on=["subject", "session"], how="inner")
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_session_skill_vs_reliability(
    df: pd.DataFrame,
    output_path: str,
    exclude_low_level: bool = False,
) -> plt.Figure:
    """Multi-panel scatter: session skill (x) vs brain reliability (y).

    One panel per condition, hued by subject, with per-subject trend
    lines (seaborn regplot).

    Args:
        df: Merged dataframe with ``skill``, ``reliability``,
            ``condition``, ``subject``.
        output_path: Where to save the figure.
        exclude_low_level: If True, exclude low-level conditions.

    Returns:
        The matplotlib Figure.
    """
    conditions = sorted(df["condition"].unique())
    if exclude_low_level:
        conditions = [c for c in conditions if c not in LOW_LEVEL_CONDITIONS]

    n_conds = len(conditions)
    if n_conds == 0:
        print("No conditions to plot.")
        return plt.figure()

    n_cols = min(4, n_conds)
    n_rows = math.ceil(n_conds / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        dpi=300,
        squeeze=False,
    )

    subjects = sorted(df["subject"].unique())
    palette = dict(zip(subjects, sns.color_palette("Set2", n_colors=len(subjects))))

    for ax_idx, cond in enumerate(conditions):
        row = ax_idx // n_cols
        col = ax_idx % n_cols
        ax = axes[row, col]

        cond_df = df[df["condition"] == cond]

        for sub in subjects:
            sub_df = cond_df[cond_df["subject"] == sub]
            if sub_df.empty:
                continue

            ax.scatter(
                sub_df["skill"],
                sub_df["reliability"],
                color=palette[sub],
                s=40,
                alpha=0.7,
                label=sub,
                zorder=3,
                edgecolors="white",
                linewidth=0.5,
            )

            if len(sub_df) >= 3:
                sns.regplot(
                    x="skill",
                    y="reliability",
                    data=sub_df,
                    scatter=False,
                    color=palette[sub],
                    ax=ax,
                    ci=None,
                    line_kws={"linewidth": 1.2, "alpha": 0.6},
                )

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel("Session skill (composite z)", fontsize=9)
        ax.set_ylabel("Session reliability (mean r)", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for ax_idx in range(n_conds, n_rows * n_cols):
        row = ax_idx // n_cols
        col = ax_idx % n_cols
        axes[row, col].set_visible(False)

    # Shared legend
    handles, labels = [], []
    for sub in subjects:
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[sub], markersize=8)
        )
        labels.append(sub)
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(subjects),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Per-session skill vs. brain-map reliability",
        fontsize=14,
        y=1.01,
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
        description="Plot per-session game skill vs. brain-map reliability."
    )
    parser.add_argument(
        "--skill-input",
        default=None,
        help="Path to session_skill_metrics.json (default: auto from config)",
    )
    parser.add_argument(
        "--corr-input",
        default=None,
        help="Path to beta_maps_correlations.pkl (default: auto from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: {FIG_PATH}/session_skill_vs_reliability.png)",
    )
    parser.add_argument(
        "--exclude-low-level",
        action="store_true",
        help="Exclude low-level conditions from figure",
    )
    args = parser.parse_args()

    skill_path = args.skill_input or op.join(
        DATA_PATH, "processed", "skill_metrics", "session_skill_metrics.json"
    )
    corr_path = args.corr_input or op.join(
        DATA_PATH, "processed", "beta_maps_correlations.pkl"
    )
    output_path = args.output or op.join(
        FIG_PATH, "session_skill_vs_reliability.png"
    )

    if not op.exists(skill_path):
        print(
            f"Error: Session skill metrics not found at {skill_path}\n"
            "Run  invoke behav.session-skill  first."
        )
        sys.exit(1)
    if not op.exists(corr_path):
        print(f"Error: Correlation data not found at {corr_path}")
        sys.exit(1)

    print("Loading session skill metrics...")
    skill_data = load_session_skill(skill_path)

    print("Loading beta correlations...")
    corr_data = load_beta_correlations(corr_path)

    print("Computing per-session brain reliability...")
    reliability_df = compute_per_session_reliability(corr_data, SUBJECTS)
    print(f"  {len(reliability_df)} (subject, session, condition) entries")

    print("Building merged dataframe...")
    merged = build_merged_dataframe(skill_data, reliability_df)
    print(f"  {len(merged)} rows after inner join")

    if merged.empty:
        print("Warning: No matching sessions between skill and reliability data.")
        sys.exit(0)

    print("Generating figure...")
    plot_session_skill_vs_reliability(
        merged, output_path, exclude_low_level=args.exclude_low_level
    )


if __name__ == "__main__":
    main()
