"""
Generate a descriptive table of the Shinobi dataset.

Produces per-subject, per-level statistics:
  - sessions: number of unique scanning/training sessions
  - runs: number of unique fMRI runs (fMRI source only)
  - N: number of valid repetitions (after excluding fake reps)
  - cleared: number of repetitions completed without losing a life
  - duration: total play duration formatted as h:m:s

Works with two data sources:
  - ``fmri``: gamelogs recorded during fMRI scanning sessions
  - ``training``: home training sessions

Outputs are saved as CSV and LaTeX, with a provenance sidecar.
"""

import argparse
import json
import logging
import os
import os.path as op
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shinobi_fmri.config import DATA_PATH, GAMELOGS_PATH, SUBJECTS, TABLE_PATH
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata, save_metadata_json


LEVELS = ["level-1", "level-4", "level-5"]
MIN_SCORE_THRESHOLD = 200


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def find_summary_files(
    data_root: str,
    subjects: List[str],
    source: str,
) -> List[Dict[str, str]]:
    """Discover all summary JSON files for the given data source.

    Args:
        data_root: Root directory containing subject folders.
        subjects: Subject IDs to include.
        source: ``"fmri"`` or ``"training"``, controls subdirectory layout.

    Returns:
        List of dicts with keys ``subject``, ``session``, ``level``,
        ``run`` (or ``None`` for training), and ``filepath``.
    """
    subdir = "gamelogs" if source == "fmri" else "beh"
    entries: List[Dict[str, Any]] = []

    for sub in subjects:
        sub_dir = op.join(data_root, sub)
        if not op.isdir(sub_dir):
            continue

        sessions = sorted(
            s for s in os.listdir(sub_dir) if s.startswith("ses-")
        )
        for ses in sessions:
            ses_dir = op.join(sub_dir, ses, subdir)
            if not op.isdir(ses_dir):
                continue

            for fname in sorted(os.listdir(ses_dir)):
                if not fname.endswith("_summary.json"):
                    continue

                level = _extract_level(fname)
                if level is None:
                    continue

                run = _extract_run(fname)

                entries.append({
                    "subject": sub,
                    "session": ses,
                    "level": level,
                    "run": run,
                    "filepath": op.join(ses_dir, fname),
                })

    return entries


def _extract_level(filename: str) -> Optional[str]:
    """Extract the level identifier from a filename.

    Args:
        filename: Basename of the summary file.

    Returns:
        Level string (e.g. ``"level-1"``) or ``None`` if not matched.
    """
    match = re.search(r"(level-\d+)", filename)
    if match and match.group(1) in LEVELS:
        return match.group(1)
    return None


def _extract_run(filename: str) -> Optional[str]:
    """Extract the run identifier from a filename.

    Args:
        filename: Basename of the summary file.

    Returns:
        Run string (e.g. ``"run-01"``) or ``None`` if not present.
    """
    match = re.search(r"(run-\d+)", filename)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Summary reading and filtering
# ---------------------------------------------------------------------------

def load_summary(filepath: str) -> Dict[str, Any]:
    """Load a ``*_summary.json`` file.

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        Parsed summary dict.
    """
    with open(filepath) as f:
        return json.load(f)


def is_valid_rep(summary: Dict[str, Any]) -> bool:
    """Check whether a repetition is valid (not a fake rep).

    Fake reps have an end score at or below ``MIN_SCORE_THRESHOLD``.

    Args:
        summary: Parsed summary dict from a ``_summary.json`` file.

    Returns:
        ``True`` if the repetition should be included.
    """
    return summary.get("end_score", 0) > MIN_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------

def format_duration(total_seconds: float) -> str:
    """Format seconds as ``hh:mm:ss``.

    Args:
        total_seconds: Duration in seconds.

    Returns:
        Formatted string.
    """
    total_seconds = int(round(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def compute_table(
    entries: List[Dict[str, Any]],
    subjects: List[str],
    levels: List[str],
    source: str,
    logger: AnalysisLogger,
) -> pd.DataFrame:
    """Compute the descriptive table from summary files.

    Args:
        entries: File entries from :func:`find_summary_files`.
        subjects: Subject IDs (row order).
        levels: Level identifiers (column group order).
        source: ``"fmri"`` or ``"training"`` (controls whether runs
            column is included).
        logger: Logger instance.

    Returns:
        DataFrame with multi-level columns
        ``(level, metric)`` and rows per subject + Total.
    """
    has_runs = source == "fmri"

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sub in subjects:
        stats[sub] = {}
        for lv in levels:
            stats[sub][lv] = {
                "n": 0,
                "cleared": 0,
                "duration": 0.0,
                "sessions": set(),
                "runs": set(),
            }

    skipped = 0
    included = 0

    for entry in entries:
        sub = entry["subject"]
        lv = entry["level"]
        if sub not in stats:
            continue

        summary = load_summary(entry["filepath"])

        if not is_valid_rep(summary):
            skipped += 1
            continue

        included += 1
        stats[sub][lv]["n"] += 1
        stats[sub][lv]["sessions"].add(entry["session"])
        if entry["run"] is not None:
            stats[sub][lv]["runs"].add((entry["session"], entry["run"]))
        if summary.get("cleared", False):
            stats[sub][lv]["cleared"] += 1
        stats[sub][lv]["duration"] += summary.get("duration", 0.0)

    logger.info(f"Included {included} repetitions, skipped {skipped} fake reps")

    level_metrics = ["N", "cleared", "duration (h:m:s)"]
    total_metrics = ["sessions"]
    if has_runs:
        total_metrics.append("runs")
    total_metrics.extend(level_metrics)

    rows = []
    for sub in subjects:
        row: Dict[Tuple[str, str], Any] = {}
        total_n = 0
        total_cleared = 0
        total_duration = 0.0
        total_sessions: set = set()
        total_runs: set = set()

        for lv in levels:
            s = stats[sub][lv]
            row[(lv, "N")] = int(s["n"])
            row[(lv, "cleared")] = int(s["cleared"])
            row[(lv, "duration (h:m:s)")] = format_duration(s["duration"])
            total_n += s["n"]
            total_cleared += s["cleared"]
            total_duration += s["duration"]
            total_sessions |= s["sessions"]
            total_runs |= s["runs"]

        row[("Total", "sessions")] = len(total_sessions)
        if has_runs:
            row[("Total", "runs")] = len(total_runs)
        row[("Total", "N")] = int(total_n)
        row[("Total", "cleared")] = int(total_cleared)
        row[("Total", "duration (h:m:s)")] = format_duration(total_duration)
        rows.append(row)

    # Total row
    total_row: Dict[Tuple[str, str], Any] = {}
    grand_n = 0
    grand_cleared = 0
    grand_duration = 0.0
    grand_sessions: set = set()
    grand_runs: set = set()

    for lv in levels:
        lv_n = 0
        lv_cleared = 0
        lv_duration = 0.0
        for sub in subjects:
            lv_n += stats[sub][lv]["n"]
            lv_cleared += stats[sub][lv]["cleared"]
            lv_duration += stats[sub][lv]["duration"]

        total_row[(lv, "N")] = int(lv_n)
        total_row[(lv, "cleared")] = int(lv_cleared)
        total_row[(lv, "duration (h:m:s)")] = format_duration(lv_duration)
        grand_n += lv_n
        grand_cleared += lv_cleared
        grand_duration += lv_duration
        for sub in subjects:
            grand_sessions |= {(sub, s) for s in stats[sub][lv]["sessions"]}
            grand_runs |= {(sub, r) for r in stats[sub][lv]["runs"]}

    total_row[("Total", "sessions")] = len(grand_sessions)
    if has_runs:
        total_row[("Total", "runs")] = len(grand_runs)
    total_row[("Total", "N")] = int(grand_n)
    total_row[("Total", "cleared")] = int(grand_cleared)
    total_row[("Total", "duration (h:m:s)")] = format_duration(grand_duration)
    rows.append(total_row)

    columns = pd.MultiIndex.from_tuples(
        [(lv, m) for lv in levels for m in level_metrics]
        + [("Total", m) for m in total_metrics],
    )
    index = subjects + ["Total"]

    df = pd.DataFrame(rows, index=pd.Index(index, name="participant"), columns=columns)
    return df.T


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_table(
    df: pd.DataFrame,
    output_dir: str,
    basename: str,
    logger: AnalysisLogger,
) -> List[str]:
    """Save the table as CSV and LaTeX.

    Args:
        df: Descriptive table DataFrame.
        output_dir: Directory for outputs.
        basename: File name stem (without extension).
        logger: Logger instance.

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = op.join(output_dir, f"{basename}.csv")
    latex_path = op.join(output_dir, f"{basename}.tex")

    df.to_csv(csv_path)
    logger.info(f"CSV saved to {csv_path}")

    latex_str = df.to_latex(multicolumn=True, multirow=True)
    with open(latex_path, "w") as f:
        f.write(latex_str)
    logger.info(f"LaTeX saved to {latex_path}")

    return [csv_path, latex_path]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a descriptive table of the Shinobi dataset.",
    )
    parser.add_argument(
        "--source",
        choices=["fmri", "training"],
        default="fmri",
        help="Data source: 'fmri' for scanner gamelogs, 'training' for home training (default: fmri)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root directory containing subject folders. "
            "Default: GAMELOGS_PATH for fmri, {DATA_PATH}/shinobi_training for training."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {TABLE_PATH})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument("--log-dir", default=None, help="Custom log directory")
    args = parser.parse_args()

    verbosity = {0: logging.WARNING, 1: logging.INFO}.get(
        args.verbose, logging.DEBUG
    )
    logger = AnalysisLogger(
        log_name="behavioral_descriptive_table",
        log_dir=args.log_dir,
        verbosity=verbosity,
    )

    if args.data_root is not None:
        data_root = args.data_root
    elif args.source == "fmri":
        data_root = GAMELOGS_PATH
    else:
        data_root = op.join(DATA_PATH, "shinobi_training")

    output_dir = args.output_dir or TABLE_PATH
    basename = f"descriptive_table_{args.source}"

    logger.info(f"Source: {args.source}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")

    logger.info("Discovering summary files...")
    entries = find_summary_files(data_root, SUBJECTS, args.source)
    logger.info(f"Found {len(entries)} summary files")

    if not entries:
        logger.error(f"No summary files found under {data_root}")
        logger.close()
        sys.exit(1)

    logger.info("Computing descriptive table...")
    df = compute_table(entries, SUBJECTS, LEVELS, args.source, logger)

    logger.info("Table preview:")
    logger.info(f"\n{df.to_string()}")

    output_files = save_table(df, output_dir, basename, logger)

    metadata = create_metadata(
        description=f"Descriptive table of the Shinobi {args.source} dataset",
        script_path=__file__,
        output_files=output_files,
        parameters={
            "source": args.source,
            "data_root": data_root,
            "subjects": SUBJECTS,
            "levels": LEVELS,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
        },
    )
    save_metadata_json(
        metadata,
        op.join(output_dir, f"{basename}_metadata.json"),
    )

    logger.close()


if __name__ == "__main__":
    main()
