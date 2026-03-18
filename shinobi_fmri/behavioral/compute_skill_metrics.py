"""
Compute per-subject game skill metrics from Shinobi gamelogs.

Reads frame-by-frame game variables and summary JSON files from the BIDS
gamelogs directory and produces three skill metrics per subject:

1. **Clear rate**: proportion of gameplays where the level was completed
   without losing a life.
2. **Average progression**: mean percentage of each level reached,
   normalized by the maximum position observed across all subjects.
3. **Efficiency**: mean ratio of progression to damage taken,
   computed as ``completion_pct / (|health_lost| + 1)``.

Outputs are saved as JSON (one entry per subject) with a provenance
sidecar.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import (
    DATA_PATH,
    GAMELOGS_PATH,
    SUBJECTS,
)
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_metadata_json


LEVELS = ["level-1", "level-4", "level-5"]
MIN_SCORE_THRESHOLD = 200  # Filter out "fake" repetitions


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fix_position_resets(raw_x_player: List[int]) -> List[int]:
    """Fix X_player resets so position increases monotonically.

    The game resets X_player to 0 when the screen scrolls past a section
    boundary.  This function accumulates a correction offset so the
    returned list is continuous.

    Args:
        raw_x_player: Per-frame horizontal positions from one repetition.

    Returns:
        Corrected position list (same length minus one frame, matching
        the convention in ``shinobi_behav``).
    """
    fixed = [raw_x_player[0]]
    correction = 0
    for i in range(1, len(raw_x_player) - 1):
        if raw_x_player[i - 1] - raw_x_player[i] > 100:
            correction += raw_x_player[i - 1] - raw_x_player[i]
        fixed.append(raw_x_player[i] + correction)
    return fixed


def load_gamelog_variables(filepath: str) -> Dict[str, Any]:
    """Load a ``*_variables.json`` gamelog file.

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        Parsed dictionary with frame-by-frame game state.
    """
    with open(filepath) as f:
        return json.load(f)


def collect_gamelogs(
    gamelogs_root: str,
    subjects: List[str],
    levels: List[str],
    logger: AnalysisLogger,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Discover and load all valid gamelog variable files.

    Args:
        gamelogs_root: Root of the BIDS gamelogs tree.
        subjects: Subject IDs to process.
        levels: Level identifiers to include (e.g. ``["level-1", ...]``).
        logger: Logger instance.

    Returns:
        Nested dict ``{subject: {level: [var_dict, ...]}}``.
    """
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        sub: {lv: [] for lv in levels} for sub in subjects
    }

    for sub in subjects:
        sub_dir = op.join(gamelogs_root, sub)
        if not op.isdir(sub_dir):
            logger.warning(f"Subject directory not found: {sub_dir}")
            continue

        sessions = sorted(
            s for s in os.listdir(sub_dir) if s.startswith("ses-")
        )
        for ses in sessions:
            gamelogs_dir = op.join(sub_dir, ses, "gamelogs")
            if not op.isdir(gamelogs_dir):
                continue

            var_files = sorted(
                glob(op.join(gamelogs_dir, "*_variables.json"))
            )
            for vf in var_files:
                basename = op.basename(vf)
                level_str = next(
                    (lv for lv in levels if lv in basename), None
                )
                if level_str is None:
                    continue

                var_dict = load_gamelog_variables(vf)

                if max(var_dict.get("score", [0])) <= MIN_SCORE_THRESHOLD:
                    continue

                data[sub][level_str].append(var_dict)

    for sub in subjects:
        counts = {lv: len(data[sub][lv]) for lv in levels}
        total = sum(counts.values())
        logger.info(f"{sub}: {counts} (total: {total})")

    return data


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_clear_rate(
    reps: List[Dict[str, Any]],
) -> float:
    """Proportion of repetitions completed without losing a life.

    Args:
        reps: List of variable dicts for one subject/level combination.

    Returns:
        Clear rate in [0, 1].
    """
    if not reps:
        return 0.0
    clears = 0
    for rep in reps:
        lives_lost = sum(1 for x in np.diff(rep["lives"]) if x < 0)
        if lives_lost == 0:
            clears += 1
    return clears / len(reps)


def compute_max_positions(
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    subjects: List[str],
    levels: List[str],
) -> Dict[str, float]:
    """Find the maximum corrected X position per level across all subjects.

    The ``-100`` offset accounts for boss-fight jitter at the end of
    each level, following the convention in ``shinobi_behav``.

    Args:
        data: Nested gamelog data.
        subjects: Subject IDs.
        levels: Level identifiers.

    Returns:
        ``{level: end_of_level_position}``.
    """
    level_max: Dict[str, float] = {}
    for lv in levels:
        all_max = []
        for sub in subjects:
            for rep in data[sub][lv]:
                fixed = fix_position_resets(rep["X_player"])
                all_max.append(max(fixed))
        level_max[lv] = max(all_max) - 100 if all_max else 1.0
    return level_max


def compute_progression(
    reps: List[Dict[str, Any]],
    end_of_level: float,
) -> List[float]:
    """Percentage of the level reached in each repetition.

    Args:
        reps: Variable dicts for one subject/level.
        end_of_level: Maximum position defining 100 %.

    Returns:
        List of completion percentages (0--100).
    """
    pcts = []
    for rep in reps:
        fixed = fix_position_resets(rep["X_player"])
        max_x = min(max(fixed), end_of_level)
        pcts.append(max_x / end_of_level * 100)
    return pcts


def compute_efficiency(
    reps: List[Dict[str, Any]],
    end_of_level: float,
) -> List[float]:
    """Efficiency: progression divided by damage taken (plus one).

    Args:
        reps: Variable dicts for one subject/level.
        end_of_level: Maximum position defining 100 %.

    Returns:
        List of efficiency values per repetition.
    """
    effs = []
    for rep in reps:
        fixed = fix_position_resets(rep["X_player"])
        max_x = min(max(fixed), end_of_level)
        pct = max_x / end_of_level * 100
        health_change = np.diff(rep["health"])
        total_hlost = abs(sum(x for x in health_change if x < 0))
        effs.append(pct / (total_hlost + 1))
    return effs


def compute_all_metrics(
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    subjects: List[str],
    levels: List[str],
    logger: AnalysisLogger,
) -> Dict[str, Dict[str, Any]]:
    """Compute all three skill metrics for every subject.

    Args:
        data: Nested gamelog data.
        subjects: Subject IDs.
        levels: Level identifiers.
        logger: Logger instance.

    Returns:
        ``{subject: {metric_name: value, ...}}`` with both per-level
        breakdowns and overall aggregates.
    """
    level_max = compute_max_positions(data, subjects, levels)
    for lv, pos in level_max.items():
        logger.info(f"{lv} end-of-level position: {pos:.0f}")

    results: Dict[str, Dict[str, Any]] = {}

    for sub in subjects:
        all_pcts: List[float] = []
        all_effs: List[float] = []
        total_clears = 0
        total_attempts = 0
        per_level: Dict[str, Dict[str, float]] = {}

        for lv in levels:
            reps = data[sub][lv]
            n_reps = len(reps)

            cr = compute_clear_rate(reps)
            pcts = compute_progression(reps, level_max[lv])
            effs = compute_efficiency(reps, level_max[lv])

            clears = int(round(cr * n_reps))
            total_clears += clears
            total_attempts += n_reps
            all_pcts.extend(pcts)
            all_effs.extend(effs)

            per_level[lv] = {
                "n_reps": n_reps,
                "clear_rate": cr,
                "avg_progression": float(np.mean(pcts)) if pcts else 0.0,
                "avg_efficiency": float(np.mean(effs)) if effs else 0.0,
            }

        overall_cr = total_clears / total_attempts if total_attempts else 0.0
        overall_prog = float(np.mean(all_pcts)) if all_pcts else 0.0
        overall_eff = float(np.mean(all_effs)) if all_effs else 0.0

        results[sub] = {
            "clear_rate": overall_cr,
            "avg_progression": overall_prog,
            "avg_efficiency": overall_eff,
            "total_attempts": total_attempts,
            "per_level": per_level,
        }

        logger.info(
            f"{sub}: clear_rate={overall_cr:.3f}  "
            f"progression={overall_prog:.1f}%  "
            f"efficiency={overall_eff:.2f}  "
            f"(n={total_attempts})"
        )

    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    logger: AnalysisLogger,
) -> None:
    """Write skill metrics to JSON.

    Args:
        results: Computed metrics dict.
        output_path: Destination file path.
        logger: Logger instance.
    """
    os.makedirs(op.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-subject game skill metrics from Shinobi gamelogs."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path "
            "(default: {DATA_PATH}/processed/skill_metrics/skill_metrics.json)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
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
        log_name="behavioral_skill_metrics",
        log_dir=args.log_dir,
        verbosity=verbosity,
    )

    output_path = args.output or op.join(
        DATA_PATH, "processed", "skill_metrics", "skill_metrics.json"
    )

    logger.info("Collecting gamelogs...")
    data = collect_gamelogs(GAMELOGS_PATH, SUBJECTS, LEVELS, logger)

    logger.info("Computing skill metrics...")
    results = compute_all_metrics(data, SUBJECTS, LEVELS, logger)

    save_results(results, output_path, logger)

    metadata = create_metadata(
        description="Per-subject game skill metrics (clear rate, progression, efficiency)",
        script_path=__file__,
        output_files=[output_path],
        parameters={
            "subjects": SUBJECTS,
            "levels": LEVELS,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "gamelogs_root": GAMELOGS_PATH,
        },
    )
    save_metadata_json(metadata, output_path.replace(".json", "_metadata.json"))

    logger.close()


if __name__ == "__main__":
    main()
