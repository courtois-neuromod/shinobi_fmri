"""
Compute per-session composite game skill metrics from Shinobi gamelogs.

Reads frame-by-frame game variables from the BIDS gamelogs directory and
produces a composite skill score per repetition, then aggregates to
per-session means.

The composite metric combines three z-scored raw metrics (within each
level across all subjects):

    composite = 3 * z(progression) - 2 * z(health_lost) + 1 * z(speed)

Outputs are saved as JSON (keyed by subject then session) with a
provenance sidecar.
"""

import argparse
import json
import logging
import os
import os.path as op
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.behavioral.compute_skill_metrics import (
    collect_gamelogs,
    compute_max_positions,
    fix_position_resets,
    load_gamelog_variables,
)
from shinobi_fmri.config import (
    DATA_PATH,
    GAMELOGS_PATH,
    SUBJECTS,
)
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_metadata_json


LEVELS = ["level-1", "level-4", "level-5"]
MIN_SCORE_THRESHOLD = 200

DEFAULT_WEIGHTS = {"progression": 3.0, "health_lost": -2.0, "speed": 1.0}


# ---------------------------------------------------------------------------
# Gamelog collection (session-aware)
# ---------------------------------------------------------------------------

def collect_gamelogs_by_session(
    gamelogs_root: str,
    subjects: List[str],
    levels: List[str],
    logger: AnalysisLogger,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Discover and load valid gamelog variable files, grouped by session.

    Args:
        gamelogs_root: Root of the BIDS gamelogs tree.
        subjects: Subject IDs to process.
        levels: Level identifiers to include.
        logger: Logger instance.

    Returns:
        Nested dict ``{subject: {session: {level: [var_dict, ...]}}}``.
    """
    data: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    for sub in subjects:
        data[sub] = {}
        sub_dir = op.join(gamelogs_root, sub)
        if not op.isdir(sub_dir):
            logger.warning(f"Subject directory not found: {sub_dir}")
            continue

        sessions = sorted(
            s for s in os.listdir(sub_dir) if s.startswith("ses-")
        )
        for ses in sessions:
            data[sub][ses] = {lv: [] for lv in levels}
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

                data[sub][ses][level_str].append(var_dict)

    for sub in subjects:
        total = sum(
            len(data[sub][ses][lv])
            for ses in data[sub]
            for lv in levels
        )
        n_sessions = len(data[sub])
        logger.info(f"{sub}: {n_sessions} sessions, {total} total reps")

    return data


# ---------------------------------------------------------------------------
# Per-repetition raw metrics
# ---------------------------------------------------------------------------

def compute_rep_raw_metrics(
    rep: Dict[str, Any],
    end_of_level: float,
) -> Dict[str, float]:
    """Compute raw metrics for a single repetition.

    Args:
        rep: Variable dict for one repetition.
        end_of_level: Maximum position defining 100%.

    Returns:
        Dict with keys ``progression``, ``health_lost``, ``speed``,
        ``n_frames``.
    """
    fixed = fix_position_resets(rep["X_player"])
    max_x = min(max(fixed), end_of_level)
    progression = max_x / end_of_level * 100

    health_change = np.diff(rep["health"])
    health_lost = abs(sum(x for x in health_change if x < 0))

    n_frames = len(rep["X_player"])
    speed = progression / n_frames if n_frames > 0 else 0.0

    return {
        "progression": float(progression),
        "health_lost": float(health_lost),
        "speed": float(speed),
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# Composite skill (z-score + weighted sum)
# ---------------------------------------------------------------------------

def compute_composite_skill(
    all_reps: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Z-score raw metrics within each level, then combine into composite.

    Z-scoring is performed across all subjects within each level so that
    level difficulty is normalised.  The composite is:

        weights["progression"] * z(prog)
      + weights["health_lost"] * z(health_lost)
      + weights["speed"]       * z(speed)

    Default weights: ``3*z(prog) - 2*z(health_lost) + 1*z(speed)``.

    Args:
        all_reps: List of dicts, each with keys ``subject``, ``session``,
            ``level``, ``progression``, ``health_lost``, ``speed``.
        weights: Optional override for metric weights.

    Returns:
        Same list of dicts with added keys ``z_progression``,
        ``z_health_lost``, ``z_speed``, ``composite``.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    levels = sorted(set(r["level"] for r in all_reps))

    for level in levels:
        level_reps = [r for r in all_reps if r["level"] == level]
        if len(level_reps) < 2:
            for r in level_reps:
                r["z_progression"] = 0.0
                r["z_health_lost"] = 0.0
                r["z_speed"] = 0.0
            continue

        for metric in ("progression", "health_lost", "speed"):
            values = np.array([r[metric] for r in level_reps])
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            if std < 1e-12:
                std = 1.0
            for r in level_reps:
                r[f"z_{metric}"] = float((r[metric] - mean) / std)

    for r in all_reps:
        r["composite"] = (
            weights["progression"] * r["z_progression"]
            + weights["health_lost"] * r["z_health_lost"]
            + weights["speed"] * r["z_speed"]
        )

    return all_reps


# ---------------------------------------------------------------------------
# Aggregation to session level
# ---------------------------------------------------------------------------

def aggregate_per_session(
    all_reps: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute mean composite skill per (subject, session).

    Args:
        all_reps: List of rep dicts (must include ``composite`` key).

    Returns:
        ``{subject: {session: {mean_composite, n_reps, ...}}}``.
    """
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in all_reps:
        sub = r["subject"]
        ses = r["session"]
        grouped.setdefault(sub, {}).setdefault(ses, []).append(r["composite"])

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sub in sorted(grouped):
        results[sub] = {}
        for ses in sorted(grouped[sub]):
            vals = grouped[sub][ses]
            results[sub][ses] = {
                "mean_composite": float(np.mean(vals)),
                "std_composite": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_reps": len(vals),
            }

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_all_reps(
    session_data: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    level_max: Dict[str, float],
    levels: List[str],
    logger: AnalysisLogger,
) -> List[Dict[str, Any]]:
    """Build flat list of per-rep metrics from session-grouped gamelogs.

    Args:
        session_data: Output of ``collect_gamelogs_by_session``.
        level_max: End-of-level positions per level.
        levels: Level identifiers.
        logger: Logger instance.

    Returns:
        Flat list of rep dicts with subject, session, level, and raw
        metrics.
    """
    all_reps: List[Dict[str, Any]] = []
    for sub in sorted(session_data):
        for ses in sorted(session_data[sub]):
            for lv in levels:
                for rep in session_data[sub][ses][lv]:
                    metrics = compute_rep_raw_metrics(rep, level_max[lv])
                    metrics["subject"] = sub
                    metrics["session"] = ses
                    metrics["level"] = lv
                    all_reps.append(metrics)

    logger.info(f"Total repetitions: {len(all_reps)}")
    return all_reps


def save_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: str,
    logger: AnalysisLogger,
) -> None:
    """Write session-level skill metrics to JSON.

    Args:
        results: Aggregated session skill dict.
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
        description="Compute per-session composite game skill metrics."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path "
            "(default: {DATA_PATH}/processed/skill_metrics/session_skill_metrics.json)"
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
        log_name="behavioral_session_skill",
        log_dir=args.log_dir,
        verbosity=verbosity,
    )

    output_path = args.output or op.join(
        DATA_PATH, "processed", "skill_metrics", "session_skill_metrics.json"
    )

    # Step 1: collect gamelogs grouped by session
    logger.info("Collecting gamelogs by session...")
    session_data = collect_gamelogs_by_session(
        GAMELOGS_PATH, SUBJECTS, LEVELS, logger
    )

    # We also need the flat collect_gamelogs for computing max positions
    logger.info("Computing end-of-level positions...")
    flat_data = collect_gamelogs(GAMELOGS_PATH, SUBJECTS, LEVELS, logger)
    level_max = compute_max_positions(flat_data, SUBJECTS, LEVELS)
    for lv, pos in level_max.items():
        logger.info(f"{lv} end-of-level position: {pos:.0f}")

    # Step 2: build per-rep raw metrics
    logger.info("Computing per-repetition raw metrics...")
    all_reps = build_all_reps(session_data, level_max, LEVELS, logger)

    # Step 3: z-score and composite
    logger.info("Computing composite skill (z-scored)...")
    all_reps = compute_composite_skill(all_reps)

    # Step 4: aggregate per session
    logger.info("Aggregating per session...")
    results = aggregate_per_session(all_reps)

    for sub in sorted(results):
        for ses in sorted(results[sub]):
            entry = results[sub][ses]
            logger.info(
                f"{sub} {ses}: composite={entry['mean_composite']:.3f} "
                f"(n={entry['n_reps']})"
            )

    save_results(results, output_path, logger)

    metadata = create_metadata(
        description="Per-session composite game skill metrics (z-scored progression, health, speed)",
        script_path=__file__,
        output_files=[output_path],
        parameters={
            "subjects": SUBJECTS,
            "levels": LEVELS,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "gamelogs_root": GAMELOGS_PATH,
            "weights": DEFAULT_WEIGHTS,
        },
    )
    save_metadata_json(metadata, output_path.replace(".json", "_metadata.json"))

    logger.close()


if __name__ == "__main__":
    main()
