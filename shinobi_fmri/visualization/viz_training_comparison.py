"""
Compare behavioral performance between home training and scanner sessions.

Loads game replay data from both the shinobi_training (home practice) and
shinobi (scanner) datasets, computes four performance metrics per repetition,
and generates three figure panels:

1. **Matched comparison** (point plot): The last M training repetitions vs
   all scanner repetitions (where M = number of scan reps for that
   subject/level).  The most direct test of whether in-scanner performance
   matches the plateau reached at the end of training.  Includes a
   permutation test per subject/level/metric with significance brackets.

2. **Setup comparison** (point plot): Metrics split by training time window
   (0-1 week, 1-12 weeks, 12+ weeks) and scanner sessions, per subject and
   level.  Shows the full training trajectory context.

3. **Learning curves** (line plot): Smoothed metric trajectories over days of
   training, per subject and level.

Metrics:
- **Final score**: End-of-repetition game score.
- **Proportion cleared**: Percentage of the level reached, computed from
  corrected X_player position (see ``fix_position_resets``).
- **Health loss**: Total health points lost (negative = more damage).
- **Cleared**: Whether the level was completed (1) or not (0).

Statistical test:
- Two-sided permutation test (10 000 shuffles) comparing training tail vs
  scan for each subject/level/metric.  No correction for multiple comparisons.

Usage:
    python viz_training_comparison.py -v
    python viz_training_comparison.py --smoothing-window 20 -vv
    python viz_training_comparison.py --n-permutations 5000 -v
"""

import argparse
import json
import logging
import os
import os.path as op
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scipy.stats import ttest_ind

from shinobi_fmri.config import (
    FIG_PATH,
    GAMELOGS_PATH,
    SUBJECTS,
    TABLE_PATH,
    TRAINING_GAMELOGS_PATH,
)
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_metadata_json


LEVELS = ["level-1", "level-4", "level-5"]
MIN_SCORE_THRESHOLD = 200
DEFAULT_SMOOTHING_WINDOW = 15
DEFAULT_N_PERMUTATIONS = 10000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def fix_position_resets(raw_x_player: List[int]) -> List[int]:
    """Fix X_player resets so position increases monotonically.

    The game resets X_player to 0 when the screen scrolls past a section
    boundary.  This function accumulates a correction offset so the
    returned list is continuous.

    Args:
        raw_x_player: Per-frame horizontal positions from one repetition.

    Returns:
        Corrected position list (same length minus one frame).
    """
    fixed = [raw_x_player[0]]
    correction = 0
    for i in range(1, len(raw_x_player) - 1):
        if raw_x_player[i - 1] - raw_x_player[i] > 100:
            correction += raw_x_player[i - 1] - raw_x_player[i]
        fixed.append(raw_x_player[i] + correction)
    return fixed


def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents.

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(filepath) as f:
        return json.load(f)


def discover_repetitions(
    gamelogs_root: str,
    subjects: List[str],
    levels: List[str],
    setup: str,
    logger: AnalysisLogger,
) -> List[Dict[str, str]]:
    """Find all valid repetition file sets in a dataset.

    Each repetition is identified by its ``_summary.json`` file.  The function
    also checks for the corresponding ``_variables.json`` (required for
    progression) and ``_gamedata.json`` (optional, for timestamps).

    Args:
        gamelogs_root: Root directory of the dataset.
        subjects: Subject IDs to include.
        levels: Level identifiers to include.
        setup: ``"Train"`` or ``"Scan"`` label for the dataset.
        logger: Logger instance.

    Returns:
        List of dicts with keys ``subject``, ``level``, ``setup``,
        ``summary_path``, ``variables_path``, and ``gamedata_path``
        (None if not available).
    """
    reps: List[Dict[str, str]] = []

    for sub in subjects:
        sub_dir = op.join(gamelogs_root, sub)
        if not op.isdir(sub_dir):
            logger.warning(f"Subject directory not found: {sub_dir}")
            continue

        sessions = sorted(
            s for s in os.listdir(sub_dir) if s.startswith("ses-")
        )
        for ses in sessions:
            # Training data uses beh/, scan data uses gamelogs/
            beh_dir = op.join(sub_dir, ses, "beh")
            gamelogs_dir = op.join(sub_dir, ses, "gamelogs")
            data_dir = beh_dir if op.isdir(beh_dir) else gamelogs_dir
            if not op.isdir(data_dir):
                continue

            summary_files = sorted(glob(op.join(data_dir, "*_summary.json")))
            for sf in summary_files:
                basename = op.basename(sf)
                level_str = next(
                    (lv for lv in levels if lv in basename), None
                )
                if level_str is None:
                    continue

                variables_path = sf.replace("_summary.json", "_variables.json")
                if not op.isfile(variables_path):
                    continue

                gamedata_path = sf.replace("_summary.json", "_gamedata.json")
                if not op.isfile(gamedata_path):
                    gamedata_path = None

                reps.append({
                    "subject": sub,
                    "level": level_str,
                    "setup": setup,
                    "summary_path": sf,
                    "variables_path": variables_path,
                    "gamedata_path": gamedata_path,
                })

    logger.info(f"{setup}: found {len(reps)} repetitions across {len(subjects)} subjects")
    return reps


def compute_max_positions(
    all_reps: List[Dict[str, str]],
    levels: List[str],
    logger: AnalysisLogger,
) -> Dict[str, float]:
    """Find the maximum corrected X position per level across all repetitions.

    The ``-100`` offset accounts for boss-fight jitter at the end of each
    level, following the convention in ``shinobi_behav``.

    Args:
        all_reps: All discovered repetition dicts.
        levels: Level identifiers.
        logger: Logger instance.

    Returns:
        ``{level: end_of_level_position}``.
    """
    level_max: Dict[str, float] = {lv: 0.0 for lv in levels}

    for rep in all_reps:
        lv = rep["level"]
        variables = load_json(rep["variables_path"])
        fixed = fix_position_resets(variables["X_player"])
        max_pos = max(fixed) if fixed else 0.0
        if max_pos > level_max[lv]:
            level_max[lv] = max_pos

    for lv in levels:
        level_max[lv] = max(level_max[lv] - 100, 1.0)
        logger.info(f"{lv} end-of-level position: {level_max[lv]:.0f}")

    return level_max


def build_dataframe(
    all_reps: List[Dict[str, str]],
    level_max: Dict[str, float],
    logger: AnalysisLogger,
) -> pd.DataFrame:
    """Build a DataFrame with one row per repetition and all metrics.

    Args:
        all_reps: All discovered repetition dicts.
        level_max: End-of-level positions per level.
        logger: Logger instance.

    Returns:
        DataFrame with columns: Subject, Level, Setup, Final score,
        Proportion cleared, Health loss, Cleared, Timestamp.
    """
    rows: List[Dict[str, Any]] = []
    n_skipped = 0

    for rep in all_reps:
        summary = load_json(rep["summary_path"])
        variables = load_json(rep["variables_path"])

        score = summary.get("end_score", 0)
        if score <= MIN_SCORE_THRESHOLD and rep["setup"] == "Train":
            n_skipped += 1
            continue

        # Progression
        fixed = fix_position_resets(variables["X_player"])
        end_of_level = level_max[rep["level"]]
        max_x = min(max(fixed), end_of_level) if fixed else 0.0
        proportion_cleared = max_x / end_of_level * 100

        # Health loss
        health_lost = -summary.get("total health lost", 0)

        # Cleared
        cleared = int(summary.get("cleared", False))

        # Timestamp (training only)
        timestamp = np.nan
        if rep["gamedata_path"] is not None:
            gamedata = load_json(rep["gamedata_path"])
            timestamp = gamedata.get("LevelStartTimestamp", np.nan)

        rows.append({
            "Subject": rep["subject"],
            "Level": rep["level"],
            "Setup": rep["setup"],
            "Final score": score,
            "Proportion cleared": proportion_cleared,
            "Health loss": health_lost,
            "Cleared": cleared,
            "Timestamp": timestamp,
        })

    if n_skipped:
        logger.info(f"Skipped {n_skipped} training reps with score <= {MIN_SCORE_THRESHOLD}")

    df = pd.DataFrame(rows)
    logger.info(f"Built DataFrame: {len(df)} rows ({df['Setup'].value_counts().to_dict()})")
    return df


def compute_days_of_training(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days of training relative to each subject's first session.

    Args:
        df: DataFrame with Timestamp column.

    Returns:
        DataFrame with added Days of training column.
    """
    df["Days of training"] = np.nan

    for subject in df["Subject"].unique():
        train_mask = (df["Subject"] == subject) & (df["Setup"] == "Train")
        timestamps = df.loc[train_mask, "Timestamp"]
        if timestamps.empty or timestamps.isna().all():
            continue
        first_ts = timestamps.min()
        df.loc[train_mask, "Days of training"] = (timestamps - first_ts) / 86400

    return df


def compute_setup_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign training repetitions to time-based windows.

    Windows:
    - Train (0-1w): first 14 days
    - Train (1-12w): 14 to 84 days
    - Train (12w+): after 84 days
    - Scan: scanner sessions (unchanged)

    Args:
        df: DataFrame with Days of training column.

    Returns:
        DataFrame with added Setup split column.
    """
    df["Setup split"] = df["Setup"]

    train_mask = df["Setup"] == "Train"
    days = df.loc[train_mask, "Days of training"]

    df.loc[train_mask & (days < 14), "Setup split"] = "Train (0-1w)"
    df.loc[train_mask & (days >= 14) & (days < 84), "Setup split"] = "Train (1-12w)"
    df.loc[train_mask & (days >= 84), "Setup split"] = "Train (12w+)"

    return df


def tag_matched_training(
    df: pd.DataFrame,
    logger: AnalysisLogger,
) -> pd.DataFrame:
    """Tag the last M training reps, matched to scan count per subject/level.

    For each subject/level, counts the number of scan repetitions and takes
    the same number from the tail of training (sorted by timestamp).  This
    ensures equal sample sizes between groups.

    Adds a ``Tail group`` column with values ``"Training (matched)"`` for
    the tail, ``"Scan"`` for scanner reps, or ``NaN`` otherwise.

    Args:
        df: DataFrame with Timestamp column.
        logger: Logger instance.

    Returns:
        DataFrame with added Tail group column.
    """
    df["Tail group"] = np.nan
    df.loc[df["Setup"] == "Scan", "Tail group"] = "Scan"

    for sub in df["Subject"].unique():
        for lv in df["Level"].unique():
            n_scan = (
                (df["Subject"] == sub)
                & (df["Level"] == lv)
                & (df["Setup"] == "Scan")
            ).sum()

            train_mask = (
                (df["Subject"] == sub)
                & (df["Level"] == lv)
                & (df["Setup"] == "Train")
            )
            train_idx = df.loc[train_mask].sort_values("Timestamp").index
            n_available = len(train_idx)
            n_take = min(n_scan, n_available)

            if n_take == 0:
                logger.debug(f"{sub}/{lv}: no scan or training reps to match")
                continue

            tail_idx = train_idx[-n_take:]
            df.loc[tail_idx, "Tail group"] = "Training (matched)"
            logger.debug(
                f"{sub}/{lv}: matched {n_take} training reps to {n_scan} scan reps"
            )

    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    rng: np.random.Generator = None,
) -> float:
    """Two-sided permutation test for difference in means.

    Shuffles group labels ``n_permutations`` times, computing the absolute
    difference in means each time to build a null distribution.  Returns
    the proportion of permuted differences that are at least as extreme as
    the observed difference.

    Args:
        group_a: Observations from group A.
        group_b: Observations from group B.
        n_permutations: Number of permutation shuffles.
        rng: Numpy random generator (for reproducibility).

    Returns:
        Two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    observed_diff = abs(np.mean(group_a) - np.mean(group_b))
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if perm_diff >= observed_diff:
            count += 1

    return count / n_permutations


def compute_comparison_stats(
    df: pd.DataFrame,
    subjects: List[str],
    n_permutations: int,
    logger: AnalysisLogger,
) -> Dict[Tuple[str, str, str], float]:
    """Run permutation tests for all subject/level/metric combinations.

    Args:
        df: DataFrame filtered to Tail group rows only.
        subjects: Subject IDs.
        n_permutations: Number of permutation shuffles.
        logger: Logger instance.

    Returns:
        Dict mapping ``(subject, level, metric)`` to p-value.
    """
    tail_df = df.dropna(subset=["Tail group"])
    rng = np.random.default_rng(RANDOM_SEED)
    results: Dict[Tuple[str, str, str], float] = {}

    for sub in subjects:
        for lv in LEVELS:
            train_rows = tail_df[
                (tail_df["Subject"] == sub)
                & (tail_df["Level"] == lv)
                & (tail_df["Tail group"] == "Training (matched)")
            ]
            scan_rows = tail_df[
                (tail_df["Subject"] == sub)
                & (tail_df["Level"] == lv)
                & (tail_df["Tail group"] == "Scan")
            ]
            if train_rows.empty or scan_rows.empty:
                continue

            for metric in METRICS:
                a = train_rows[metric].values.astype(float)
                b = scan_rows[metric].values.astype(float)
                p = permutation_test(a, b, n_permutations, rng)
                results[(sub, lv, metric)] = p
                logger.debug(
                    f"{sub}/{lv}/{metric}: n_train={len(a)}, n_scan={len(b)}, p={p:.4f}"
                )

    return results


def p_to_stars(p: float) -> str:
    """Convert a p-value to significance stars.

    Args:
        p: p-value.

    Returns:
        Star string: ``"***"`` (p<0.001), ``"**"`` (p<0.01),
        ``"*"`` (p<0.05), or ``"n.s."`` (p>=0.05).
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values (NaN-free).

    Returns:
        Array of adjusted q-values (same order as input), capped at 1.
    """
    n = len(p_values)
    if n == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_out = np.empty(n)
    q_out[order] = np.minimum(q, 1.0)
    return q_out


def run_ttest(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Welch t-test for difference in means.

    Args:
        group_a: Observations from group A.
        group_b: Observations from group B.

    Returns:
        ``(t_statistic, p_value)``.  Returns ``(nan, nan)`` if either
        group has fewer than 2 observations.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    res = ttest_ind(group_a, group_b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def compute_ttest_stats(
    df: pd.DataFrame,
    subjects: List[str],
    logger: AnalysisLogger,
) -> Tuple[Dict[Tuple[str, str, str], float], pd.DataFrame]:
    """Run two-sided Welch t-tests and apply BH FDR correction across all tests.

    Tests are run for every (subject, level, metric) combination comparing
    ``"Training (matched)"`` vs ``"Scan"`` in the ``Tail group`` column.
    FDR correction is applied globally across all tests.

    Args:
        df: DataFrame with ``Tail group`` column populated.
        subjects: Subject IDs.
        logger: Logger instance.

    Returns:
        Tuple of:
        - q_dict: mapping ``(subject, level, metric)`` → FDR-adjusted q-value.
        - stats_df: Full results table (one row per test).
    """
    tail_df = df.dropna(subset=["Tail group"])
    records: List[Dict[str, Any]] = []

    for sub in subjects:
        for lv in LEVELS:
            train_rows = tail_df[
                (tail_df["Subject"] == sub)
                & (tail_df["Level"] == lv)
                & (tail_df["Tail group"] == "Training (matched)")
            ]
            scan_rows = tail_df[
                (tail_df["Subject"] == sub)
                & (tail_df["Level"] == lv)
                & (tail_df["Tail group"] == "Scan")
            ]
            if train_rows.empty or scan_rows.empty:
                continue

            for metric in METRICS:
                a = train_rows[metric].values.astype(float)
                b = scan_rows[metric].values.astype(float)
                t_stat, p_val = run_ttest(a, b)
                records.append({
                    "subject": sub,
                    "level": lv,
                    "metric": metric,
                    "n_train": len(a),
                    "n_scan": len(b),
                    "mean_train": round(float(np.mean(a)), 4),
                    "mean_scan": round(float(np.mean(b)), 4),
                    "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                    "p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
                })

    if not records:
        logger.warning("No records for t-test computation")
        return {}, pd.DataFrame()

    stats_df = pd.DataFrame(records)

    # BH FDR correction across all valid tests
    valid = stats_df["p_value"].notna()
    q_arr = np.full(len(stats_df), np.nan)
    q_arr[valid.values] = fdr_bh(stats_df.loc[valid, "p_value"].values)
    stats_df["q_value"] = q_arr
    stats_df["significance"] = stats_df["q_value"].apply(
        lambda q: p_to_stars(q) if not np.isnan(q) else ""
    )

    q_dict: Dict[Tuple[str, str, str], float] = {
        (row["subject"], row["level"], row["metric"]): row["q_value"]
        for _, row in stats_df.iterrows()
    }
    n_sig = (stats_df["q_value"] < 0.05).sum()
    logger.info(f"T-tests + BH FDR: {n_sig}/{len(stats_df)} significant (q<0.05)")

    return q_dict, stats_df


def save_stats_table(
    stats_df: pd.DataFrame,
    output_dir: str,
    basename: str,
    logger: AnalysisLogger,
) -> List[str]:
    """Save a stats results table as CSV and LaTeX.

    Args:
        stats_df: DataFrame with test results.
        output_dir: Directory to save tables.
        basename: File name stem (without extension).
        logger: Logger instance.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = op.join(output_dir, f"{basename}.csv")
    tex_path = op.join(output_dir, f"{basename}.tex")
    stats_df.to_csv(csv_path, index=False)
    stats_df.to_latex(tex_path, index=False, float_format="%.4f", na_rep="—")
    logger.info(f"Stats table saved: {csv_path}")
    return [csv_path, tex_path]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


METRICS = ["Final score", "Proportion cleared", "Health loss", "Cleared"]
METRIC_LIMITS = {
    "Final score": (0, 80000),
    "Proportion cleared": (20, 100),
    "Health loss": (-10, 0),
    "Cleared": (-0.1, 1.1),
}
SETUP_ORDER = ["Train (0-1w)", "Train (1-12w)", "Train (12w+)", "Scan"]

HALVES_ORDER = ["Train (first)", "Train (last)", "Scan (first)", "Scan (last)"]
HALVES_PAIRS: List[Tuple[str, str, str]] = [
    ("Train (first)", "Train (last)", "train_improvement"),
    ("Train (last)",  "Scan (first)", "transfer"),
    ("Scan (first)",  "Scan (last)",  "scan_adaptation"),
]


def tag_halves_split(df: pd.DataFrame, logger: AnalysisLogger) -> pd.DataFrame:
    """Split training and scan reps into first / last halves per subject/level.

    Training reps are ordered by timestamp; scan reps by their discovery
    order (sessions are already sorted chronologically by
    ``discover_repetitions``).

    Adds a ``Halves group`` column with values ``"Train (first)"``,
    ``"Train (last)"``, ``"Scan (first)"``, ``"Scan (last)"``, or NaN.

    Args:
        df: DataFrame with Timestamp column.
        logger: Logger instance.

    Returns:
        DataFrame with added ``Halves group`` column.
    """
    df["Halves group"] = np.nan

    for sub in df["Subject"].unique():
        for lv in df["Level"].unique():
            # Training halves – order by timestamp
            train_mask = (
                (df["Subject"] == sub)
                & (df["Level"] == lv)
                & (df["Setup"] == "Train")
            )
            train_idx = df.loc[train_mask].sort_values("Timestamp").index
            n = len(train_idx)
            if n >= 2:
                mid = n // 2
                df.loc[train_idx[:mid], "Halves group"] = "Train (first)"
                df.loc[train_idx[mid:], "Halves group"] = "Train (last)"
                logger.debug(f"{sub}/{lv} train halves: first={mid}, last={n - mid}")

            # Scan halves – use row order (session order preserved from loading)
            scan_mask = (
                (df["Subject"] == sub)
                & (df["Level"] == lv)
                & (df["Setup"] == "Scan")
            )
            scan_idx = df.loc[scan_mask].index
            n = len(scan_idx)
            if n >= 2:
                mid = n // 2
                df.loc[scan_idx[:mid], "Halves group"] = "Scan (first)"
                df.loc[scan_idx[mid:], "Halves group"] = "Scan (last)"
                logger.debug(f"{sub}/{lv} scan halves: first={mid}, last={n - mid}")

    return df


def compute_halves_stats(
    df: pd.DataFrame,
    subjects: List[str],
    apply_fdr: bool,
    logger: AnalysisLogger,
) -> Tuple[Dict[Tuple[str, str, str, str], float], pd.DataFrame]:
    """Run t-tests for all neighbouring halves pairs, optionally with BH FDR.

    Pairs tested per (subject, level, metric):
    - ``train_improvement``: Train (first) vs Train (last)
    - ``transfer``: Train (last) vs Scan (first)
    - ``scan_adaptation``: Scan (first) vs Scan (last)

    Args:
        df: DataFrame with ``Halves group`` column populated.
        subjects: Subject IDs.
        apply_fdr: If True, apply BH FDR correction across all tests.
        logger: Logger instance.

    Returns:
        Tuple of:
        - q_dict: ``(subject, level, metric, pair_key)`` → q-value (or raw p
          if ``apply_fdr`` is False).
        - stats_df: Full results table.
    """
    halves_df = df.dropna(subset=["Halves group"])
    records: List[Dict[str, Any]] = []

    for sub in subjects:
        for lv in LEVELS:
            for group_a_lbl, group_b_lbl, pair_key in HALVES_PAIRS:
                a_rows = halves_df[
                    (halves_df["Subject"] == sub)
                    & (halves_df["Level"] == lv)
                    & (halves_df["Halves group"] == group_a_lbl)
                ]
                b_rows = halves_df[
                    (halves_df["Subject"] == sub)
                    & (halves_df["Level"] == lv)
                    & (halves_df["Halves group"] == group_b_lbl)
                ]
                if a_rows.empty or b_rows.empty:
                    continue

                for metric in METRICS:
                    a = a_rows[metric].values.astype(float)
                    b = b_rows[metric].values.astype(float)
                    t_stat, p_val = run_ttest(a, b)
                    records.append({
                        "subject": sub,
                        "level": lv,
                        "comparison": pair_key,
                        "group_a": group_a_lbl,
                        "group_b": group_b_lbl,
                        "metric": metric,
                        "n_a": len(a),
                        "n_b": len(b),
                        "mean_a": round(float(np.mean(a)), 4),
                        "mean_b": round(float(np.mean(b)), 4),
                        "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                        "p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
                    })

    if not records:
        logger.warning("No records for halves t-test computation")
        return {}, pd.DataFrame()

    stats_df = pd.DataFrame(records)

    valid = stats_df["p_value"].notna()
    if apply_fdr:
        q_arr = np.full(len(stats_df), np.nan)
        q_arr[valid.values] = fdr_bh(stats_df.loc[valid, "p_value"].values)
        stats_df["q_value"] = q_arr
    else:
        stats_df["q_value"] = stats_df["p_value"]

    stats_df["significance"] = stats_df["q_value"].apply(
        lambda q: p_to_stars(q) if not np.isnan(q) else ""
    )

    q_dict: Dict[Tuple[str, str, str, str], float] = {
        (row["subject"], row["level"], row["metric"], row["comparison"]): row["q_value"]
        for _, row in stats_df.iterrows()
    }

    label = "BH FDR" if apply_fdr else "uncorrected"
    n_sig = (stats_df["q_value"] < 0.05).sum()
    logger.info(f"Halves t-tests ({label}): {n_sig}/{len(stats_df)} significant (q<0.05)")

    return q_dict, stats_df


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    text: str,
    color: str = "black",
    lw: float = 1.0,
    fontsize: int = 8,
) -> None:
    """Draw a horizontal bracket with a text label between two x positions.

    Args:
        ax: Matplotlib axes.
        x1: Left x position.
        x2: Right x position.
        y: Vertical position of the bracket (in data coords).
        text: Label to place above the bracket (e.g. ``"***"``).
        color: Bracket and text color.
        lw: Line width.
        fontsize: Font size for label.
    """
    tick_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y - tick_height, y, y, y - tick_height],
            lw=lw, color=color, clip_on=False)
    ax.text(
        (x1 + x2) / 2, y + tick_height * 0.5, text,
        ha="center", va="bottom", fontsize=fontsize, color=color,
    )


def add_significance_grid(
    ax: plt.Axes,
    sub: str,
    var: str,
    qvalues: Dict,
    levels: List[str],
    level_colors: Dict,
    vmax: float,
    y_lim_top: float,
) -> None:
    """Place a compact significance grid in the headroom above vmax.

    Grid layout: rows = levels (top-to-bottom), columns = pairs.
    Each cell shows colored significance stars (blank = ns, omitted).
    A small italic label at the top of each column names the comparison.

    Args:
        ax: Matplotlib axes.
        sub: Subject ID.
        var: Metric name.
        qvalues: Dict mapping (subject, level, metric, pair_key) -> q-value.
        levels: Ordered list of level names.
        level_colors: Dict mapping level name -> matplotlib color.
        vmax: Upper data limit (bottom of headroom).
        y_lim_top: Top of y-axis (top of headroom).
    """
    pair_midpoints = {
        "train_improvement": 0.5,
        "transfer": 1.5,
        "scan_adaptation": 2.5,
    }
    pair_labels = {
        "train_improvement": "Tr↑",
        "transfer": "Tr→Sc",
        "scan_adaptation": "Sc↑",
    }
    headroom = y_lim_top - vmax
    n_levels = len(levels)
    row_height = headroom / (n_levels + 1)  # extra row for pair label

    # Stars per level × pair
    for i_lv, lv in enumerate(levels):
        y = vmax + row_height * (n_levels - i_lv)
        for _, _, pair_key in HALVES_PAIRS:
            key = (sub, lv, var, pair_key)
            if key not in qvalues:
                continue
            stars = p_to_stars(qvalues[key])
            if not stars:
                continue
            x_mid = pair_midpoints[pair_key]
            is_sig = stars != "n.s."
            ax.text(
                x_mid, y, stars,
                ha="center", va="center",
                fontsize=10 if is_sig else 7,
                color=level_colors[lv] if is_sig else "gray",
                fontweight="bold" if is_sig else "normal",
            )


def plot_matched_comparison(
    df: pd.DataFrame,
    subjects: List[str],
    pvalues: Dict[Tuple[str, str, str], float],
    output_path: str,
    logger: AnalysisLogger,
) -> None:
    """Generate the matched training-tail vs scan point-plot panel with stats.

    4 rows (metrics) x N cols (subjects), with hue = level, and two
    x-axis categories: "Training (matched)" and "Scan".  Significance
    brackets (from permutation tests) are drawn per level.

    Args:
        df: Full DataFrame with Tail group column.
        subjects: Subject IDs in column order.
        pvalues: Dict mapping ``(subject, level, metric)`` to p-value.
        output_path: Path to save the figure.
        logger: Logger instance.
    """
    tail_df = df.dropna(subset=["Tail group"]).copy()
    if tail_df.empty:
        logger.warning("No data for matched comparison")
        return

    tail_order = ["Training (matched)", "Scan"]

    sns.set_theme(style="whitegrid")
    n_subs = len(subjects)
    fig = plt.figure(figsize=(3.5 * n_subs, 16))
    gs = gridspec.GridSpec(len(METRICS), n_subs)

    # Get the default color palette used by seaborn for hue="Level"
    level_palette = sns.color_palette(n_colors=len(LEVELS))
    level_colors = {lv: level_palette[i] for i, lv in enumerate(LEVELS)}

    for idx_var, var in enumerate(METRICS):
        vmin, vmax = METRIC_LIMITS[var]
        bracket_headroom = (vmax - vmin) * 0.20
        y_lim_top = vmax + bracket_headroom

        for idx_sub, sub in enumerate(subjects):
            ax = plt.subplot(gs[idx_var, idx_sub])
            sub_df = tail_df[tail_df["Subject"] == sub]

            sns.pointplot(
                x="Tail group",
                y=var,
                hue="Level",
                capsize=0.2,
                errwidth=0.8,
                order=tail_order,
                data=sub_df,
                ax=ax,
            )

            ax.set_ylim([vmin, y_lim_top])
            ax.legend().remove()

            # Add significance brackets per level
            for i_lv, lv in enumerate(LEVELS):
                key = (sub, lv, var)
                if key not in pvalues:
                    continue
                stars = p_to_stars(pvalues[key])
                # Stagger bracket heights for the 3 levels
                y_bracket = vmax + (i_lv + 0.5) * (bracket_headroom / len(LEVELS))
                # x positions: hue offsets within a grouped pointplot
                # seaborn places hue groups at x ± dodge around the category
                dodge = 0.8 / len(LEVELS)
                offset = (i_lv - (len(LEVELS) - 1) / 2) * dodge
                x_left = 0 + offset   # "Training (matched)" is at x=0
                x_right = 1 + offset  # "Scan" is at x=1
                add_significance_bracket(
                    ax, x_left, x_right, y_bracket,
                    stars, color=level_colors[lv],
                )

            if idx_var == 0 and idx_sub == n_subs - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[:3], labels=labels[:3])
            if idx_var == 0:
                ax.set_title(sub)
            if idx_var != len(METRICS) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                plt.xticks(rotation=30, ha="right")
            if idx_sub != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(op.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved matched comparison: {output_path}")


def plot_setup_comparison(
    df: pd.DataFrame,
    subjects: List[str],
    output_path: str,
    logger: AnalysisLogger,
) -> None:
    """Generate the setup comparison point-plot panel.

    4 rows (metrics) x N cols (subjects), with hue = level.

    Args:
        df: Full DataFrame with Setup split column.
        subjects: Subject IDs in column order.
        output_path: Path to save the figure.
        logger: Logger instance.
    """
    sns.set_theme(style="whitegrid")
    n_subs = len(subjects)
    fig = plt.figure(figsize=(4 * n_subs, 15))
    gs = gridspec.GridSpec(len(METRICS), n_subs)

    for idx_var, var in enumerate(METRICS):
        vmin, vmax = METRIC_LIMITS[var]
        for idx_sub, sub in enumerate(subjects):
            ax = plt.subplot(gs[idx_var, idx_sub])
            sub_df = df[df["Subject"] == sub]

            sns.pointplot(
                x="Setup split",
                y=var,
                hue="Level",
                capsize=0.2,
                errwidth=0.8,
                order=SETUP_ORDER,
                data=sub_df,
                ax=ax,
            )

            ax.set_ylim([vmin, vmax])
            ax.legend().remove()

            if idx_var == 0 and idx_sub == n_subs - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[:3], labels=labels[:3])
            if idx_var == 0:
                ax.set_title(sub)
            if idx_var != len(METRICS) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Setup")
                plt.xticks(rotation=45, ha="right")
            if idx_sub != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(op.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved setup comparison: {output_path}")


def plot_learning_curves(
    df: pd.DataFrame,
    subjects: List[str],
    output_path: str,
    smoothing_window: int,
    logger: AnalysisLogger,
) -> None:
    """Generate the learning-curves panel (training data only).

    4 rows (metrics) x N cols (subjects), line plots over days of training
    with rolling-mean smoothing.

    Args:
        df: Full DataFrame.
        subjects: Subject IDs in column order.
        output_path: Path to save the figure.
        smoothing_window: Window size for rolling mean.
        logger: Logger instance.
    """
    lcurves_df = df[df["Setup"] == "Train"].copy()
    if lcurves_df.empty:
        logger.warning("No training data available for learning curves")
        return

    # Sort by timestamp within each subject/level for proper rolling mean
    lcurves_df = lcurves_df.sort_values(["Subject", "Level", "Days of training"])

    # Cast integer columns to float before rolling mean to avoid dtype warnings
    numerical_cols = ["Days of training", "Final score", "Proportion cleared",
                      "Health loss", "Cleared"]
    for col in numerical_cols:
        lcurves_df[col] = lcurves_df[col].astype(float)

    # Apply rolling mean per subject/level
    for sub in subjects:
        for lv in LEVELS:
            mask = (lcurves_df["Subject"] == sub) & (lcurves_df["Level"] == lv)
            if mask.sum() < smoothing_window:
                continue
            for col in numerical_cols:
                lcurves_df.loc[mask, col] = (
                    lcurves_df.loc[mask, col]
                    .rolling(window=smoothing_window, min_periods=1)
                    .mean()
                )

    sns.set_theme(style="whitegrid")
    n_subs = len(subjects)
    fig = plt.figure(figsize=(4 * n_subs, 15))
    gs = gridspec.GridSpec(len(METRICS), n_subs)

    for idx_var, var in enumerate(METRICS):
        global_min = lcurves_df[var].min()
        global_max = lcurves_df[var].max()
        margin = (global_max - global_min) * 0.05 if global_max != global_min else 1
        if var == "Cleared":
            margin = 0

        for idx_sub, sub in enumerate(subjects):
            ax = plt.subplot(gs[idx_var, idx_sub])
            sub_df = lcurves_df[lcurves_df["Subject"] == sub]

            sns.lineplot(
                x="Days of training",
                y=var,
                hue="Level",
                data=sub_df,
                ax=ax,
            )

            ax.set_ylim([global_min - margin, global_max + margin])
            ax.legend().remove()

            if idx_var == 0 and idx_sub == n_subs - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[:3], labels=labels[:3])
            if idx_var == 0:
                ax.set_title(sub)
            if idx_var != len(METRICS) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if idx_sub != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(op.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curves: {output_path}")


def plot_halves_comparison(
    df: pd.DataFrame,
    subjects: List[str],
    qvalues: Dict[Tuple[str, str, str, str], float],
    output_path: str,
    logger: AnalysisLogger,
) -> None:
    """Generate the halves-split point-plot panel with a significance grid.

    Layout: 4 rows (metrics) × N cols (subjects).
    X-axis: Train (first) | Train (last) | Scan (first) | Scan (last).
    Significance shown as a compact grid in the headroom above vmax:
    rows = levels, columns = pairs (Tr↑ / Tr→Sc / Sc↑).

    Args:
        df: Full DataFrame with ``Halves group`` column.
        subjects: Subject IDs in column order.
        qvalues: Dict mapping ``(subject, level, metric, pair_key)`` → q-value.
        output_path: Path to save the figure.
        logger: Logger instance.
    """
    halves_df = df.dropna(subset=["Halves group"]).copy()
    if halves_df.empty:
        logger.warning("No data for halves comparison – skipping")
        return

    sns.set_theme(style="whitegrid")
    n_subs = len(subjects)
    fig = plt.figure(figsize=(3.5 * n_subs, 16))
    gs = gridspec.GridSpec(len(METRICS), n_subs)

    level_palette = sns.color_palette(n_colors=len(LEVELS))
    level_colors = {lv: level_palette[i] for i, lv in enumerate(LEVELS)}

    for idx_var, var in enumerate(METRICS):
        vmin, vmax = METRIC_LIMITS[var]
        headroom = (vmax - vmin) * 0.25
        y_lim_top = vmax + headroom

        for idx_sub, sub in enumerate(subjects):
            ax = plt.subplot(gs[idx_var, idx_sub])
            sub_df = halves_df[halves_df["Subject"] == sub]

            sns.pointplot(
                x="Halves group",
                y=var,
                hue="Level",
                capsize=0.2,
                errwidth=0.8,
                order=HALVES_ORDER,
                data=sub_df,
                ax=ax,
            )

            ax.set_ylim([vmin, y_lim_top])
            # Keep only ticks within the data range so none bleed into headroom
            ax.set_yticks([t for t in ax.get_yticks() if vmin <= t <= vmax])
            ax.legend().remove()

            # Thin vertical separator between Train and Scan groups
            ax.axvline(x=1.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

            add_significance_grid(
                ax, sub, var, qvalues, LEVELS, level_colors, vmax, y_lim_top,
            )

            if idx_var == 0 and idx_sub == n_subs - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[:3], labels=labels[:3])
            if idx_var == 0:
                ax.set_title(sub)
            if idx_var != len(METRICS) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                plt.xticks(rotation=30, ha="right")
            if idx_sub != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(op.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved halves comparison: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare behavioral performance between training and scanner sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f"Number of permutations for significance tests (default: {DEFAULT_N_PERMUTATIONS})",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help=f"Rolling mean window for learning curves (default: {DEFAULT_SMOOTHING_WINDOW})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for figures (default: reports/figures/training_comparison/)",
    )
    parser.add_argument(
        "--no-fdr-halves",
        action="store_true",
        default=False,
        help="Disable BH FDR correction for the halves comparison (default: FDR on)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (e.g. -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument("--log-dir", default=None, help="Custom log directory")
    args = parser.parse_args()

    verbosity = {0: logging.WARNING, 1: logging.INFO}.get(
        args.verbose, logging.DEBUG
    )
    logger = AnalysisLogger(
        log_name="viz_training_comparison",
        log_dir=args.log_dir,
        verbosity=verbosity,
    )

    output_dir = args.output_dir or op.join(FIG_PATH, "training_comparison")
    os.makedirs(output_dir, exist_ok=True)

    subjects = SUBJECTS
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Levels: {LEVELS}")
    logger.info(f"N permutations: {args.n_permutations}")
    logger.info(f"Smoothing window: {args.smoothing_window}")

    # Discover repetitions from both datasets
    logger.info("Discovering training repetitions...")
    train_reps = discover_repetitions(
        TRAINING_GAMELOGS_PATH, subjects, LEVELS, "Train", logger
    )
    logger.info("Discovering scanner repetitions...")
    scan_reps = discover_repetitions(
        GAMELOGS_PATH, subjects, LEVELS, "Scan", logger
    )
    all_reps = train_reps + scan_reps

    if not all_reps:
        logger.error("No repetitions found. Check data paths in config.yaml.")
        logger.close()
        return

    # Compute normalization constants
    logger.info("Computing max positions per level...")
    level_max = compute_max_positions(all_reps, LEVELS, logger)

    # Build DataFrame
    logger.info("Building DataFrame...")
    df = build_dataframe(all_reps, level_max, logger)

    # Compute time features
    df = compute_days_of_training(df)
    df = compute_setup_splits(df)
    df = tag_matched_training(df, logger)

    # Log summary per subject
    for sub in subjects:
        sub_df = df[df["Subject"] == sub]
        splits = sub_df["Setup split"].value_counts().to_dict()
        logger.info(f"{sub}: {splits}")

    # ── Matched comparison: t-test + BH FDR ────────────────────────────────
    logger.info("Running t-tests (Welch) with BH FDR correction...")
    q_matched, matched_stats_df = compute_ttest_stats(df, subjects, logger)
    save_stats_table(
        matched_stats_df, TABLE_PATH,
        "annexA2_matched_train_vs_scan_ttest",
        logger,
    )

    # ── Halves comparison ───────────────────────────────────────────────────
    logger.info("Tagging first/last halves...")
    df = tag_halves_split(df, logger)

    fdr_halves = not args.no_fdr_halves
    logger.info(f"Running halves t-tests (FDR={'BH' if fdr_halves else 'off'})...")
    q_halves, halves_stats_df = compute_halves_stats(df, subjects, fdr_halves, logger)
    save_stats_table(
        halves_stats_df, TABLE_PATH,
        "annexA2_halves_comparison_ttest",
        logger,
    )

    # ── Figures ─────────────────────────────────────────────────────────────
    logger.info("Generating matched comparison figure...")
    plot_matched_comparison(
        df, subjects, q_matched,
        op.join(output_dir, "matched_training_vs_scan.png"),
        logger,
    )

    logger.info("Generating halves comparison figure...")
    plot_halves_comparison(
        df, subjects, q_halves,
        op.join(output_dir, "halves_training_vs_scan.png"),
        logger,
    )

    logger.info("Generating setup comparison figure...")
    plot_setup_comparison(
        df, subjects,
        op.join(output_dir, "training_vs_scan_comparison.png"),
        logger,
    )

    logger.info("Generating learning curves figure...")
    plot_learning_curves(
        df, subjects,
        op.join(output_dir, "training_learning_curves.png"),
        args.smoothing_window,
        logger,
    )

    # ── Provenance ──────────────────────────────────────────────────────────
    fig_files = [
        op.join(output_dir, "matched_training_vs_scan.png"),
        op.join(output_dir, "halves_training_vs_scan.png"),
        op.join(output_dir, "training_vs_scan_comparison.png"),
        op.join(output_dir, "training_learning_curves.png"),
    ]
    table_files = [
        op.join(TABLE_PATH, "annexA2_matched_train_vs_scan_ttest.csv"),
        op.join(TABLE_PATH, "annexA2_halves_comparison_ttest.csv"),
    ]
    metadata = create_metadata(
        description="Training vs scanner behavioral performance comparison figures",
        script_path=__file__,
        output_files=fig_files + table_files,
        parameters={
            "subjects": subjects,
            "levels": LEVELS,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "statistical_test": "Welch t-test (two-sided)",
            "fdr_method": "Benjamini-Hochberg",
            "fdr_halves": fdr_halves,
            "random_seed": RANDOM_SEED,
            "smoothing_window": args.smoothing_window,
            "training_gamelogs": TRAINING_GAMELOGS_PATH,
            "scan_gamelogs": GAMELOGS_PATH,
        },
    )
    save_metadata_json(metadata, op.join(output_dir, "metadata.json"))

    logger.info("Done.")
    logger.close()


if __name__ == "__main__":
    main()
