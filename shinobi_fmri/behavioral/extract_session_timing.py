#!/usr/bin/env python3
"""
Extract session timing information for each participant.

For the training phase, timestamps are read from the ``LevelStartTime`` field
of ``*_gamedata.json`` files in the shinobi_training BIDS dataset.

For the scanning phase, acquisition dates are stored in ``*_scans.tsv`` files
inside the shinobi MRI dataset, which are managed by git-annex and may not be
locally available (they live in the ``mri.sensitive`` remote).  When the files
are unavailable the column is left as ``NaT`` and a warning is printed.

Output
------
Saves ``session_timing.csv`` to TABLE_PATH with columns:
    subject, phase, n_sessions, date_first, date_last, duration_days, duration_months

Usage
-----
    python extract_session_timing.py
    invoke behav.session-timing
"""

import os
import os.path as op
import json
import warnings
from glob import glob
from datetime import datetime
from typing import Optional

import pandas as pd

import shinobi_fmri.config as config


# ---------------------------------------------------------------------------
# Training timestamps
# ---------------------------------------------------------------------------

def _load_training_timestamps(training_path: str, subject: str) -> list[str]:
    """Return sorted list of LevelStartTime strings for a subject's training."""
    pattern = op.join(training_path, subject, "ses-*", "beh", "*_gamedata.json")
    timestamps = []
    for fpath in sorted(glob(pattern)):
        try:
            with open(fpath) as fh:
                data = json.load(fh)
            if "LevelStartTime" in data:
                timestamps.append(data["LevelStartTime"])
        except Exception:
            continue
    return sorted(timestamps)


def _count_training_sessions(training_path: str, subject: str) -> int:
    """Return the number of unique training sessions (ses-XXX directories)."""
    pattern = op.join(training_path, subject, "ses-*", "beh", "*_gamedata.json")
    sessions = set()
    for fpath in glob(pattern):
        ses = fpath.split("/ses-")[1].split("/")[0]
        sessions.add(ses)
    return len(sessions)


# ---------------------------------------------------------------------------
# Scan timestamps
# ---------------------------------------------------------------------------

def _load_scan_timestamps(scan_path: str, subject: str) -> list[str]:
    """Return sorted acquisition timestamps from *_scans.tsv files.

    The files are git-annex managed (sensitive MRI data) and may not be
    locally available.  Returns an empty list with a warning when unavailable.
    """
    pattern = op.join(scan_path, subject, "ses-*", f"{subject}_ses-*_scans.tsv")
    timestamps = []
    missing = 0
    for fpath in sorted(glob(pattern)):
        if not op.exists(fpath):          # git-annex symlink not resolved
            missing += 1
            continue
        try:
            df = pd.read_csv(fpath, sep="\t")
            if "acq_time" in df.columns:
                for val in df["acq_time"].dropna():
                    # acq_time format: YYYY-MM-DDTHH:MM:SS
                    timestamps.append(str(val)[:10])
        except Exception:
            continue
    if missing:
        warnings.warn(
            f"{subject}: {missing} scans.tsv file(s) unavailable (git-annex remote required)."
            " Scan dates will be missing in output.",
            UserWarning,
        )
    return sorted(timestamps)


def _count_scan_sessions(scan_path: str, subject: str) -> int:
    """Return the number of scan session directories for a subject."""
    sub_path = op.join(scan_path, subject)
    if not op.isdir(sub_path):
        return 0
    return sum(1 for d in os.listdir(sub_path) if d.startswith("ses-"))


# ---------------------------------------------------------------------------
# Summary row builder
# ---------------------------------------------------------------------------

def _build_row(
    subject: str,
    phase: str,
    n_sessions: int,
    timestamps: list[str],
) -> dict:
    """Build a single summary row dict from a list of date strings."""
    if timestamps:
        fmt = "%Y-%m-%d"
        date_first = datetime.strptime(timestamps[0][:10], fmt)
        date_last = datetime.strptime(timestamps[-1][:10], fmt)
        duration_days = (date_last - date_first).days
        duration_months = round(duration_days / 30.44, 1)
    else:
        date_first = date_last = None
        duration_days = duration_months = None

    return {
        "subject": subject,
        "phase": phase,
        "n_sessions": n_sessions,
        "date_first": timestamps[0][:10] if timestamps else None,
        "date_last": timestamps[-1][:10] if timestamps else None,
        "duration_days": duration_days,
        "duration_months": duration_months,
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_session_timing(
    subjects: list[str],
    training_path: str,
    scan_path: str,
) -> pd.DataFrame:
    """Extract session timing for all subjects and both phases.

    Args:
        subjects: List of subject IDs (e.g. ['sub-01', 'sub-02']).
        training_path: Root of the shinobi_training BIDS dataset.
        scan_path: Root of the shinobi MRI BIDS dataset.

    Returns:
        DataFrame with one row per subject × phase.
    """
    rows = []
    for subject in subjects:
        # Training phase
        train_ts = _load_training_timestamps(training_path, subject)
        n_train = _count_training_sessions(training_path, subject)
        rows.append(_build_row(subject, "training", n_train, train_ts))

        # Scan phase
        scan_ts = _load_scan_timestamps(scan_path, subject)
        n_scan = _count_scan_sessions(scan_path, subject)
        rows.append(_build_row(subject, "scanning", n_scan, scan_ts))

    return pd.DataFrame(rows)


def save_session_timing_table(df: pd.DataFrame, table_path: str) -> list[str]:
    """Save session timing table as CSV and LaTeX.

    Files written:
        - {table_path}/session_timing.csv
        - {table_path}/session_timing.tex

    Args:
        df: Output of extract_session_timing().
        table_path: Directory to save tables.

    Returns:
        List of saved file paths.
    """
    os.makedirs(table_path, exist_ok=True)
    csv_path = op.join(table_path, "session_timing.csv")
    tex_path = op.join(table_path, "session_timing.tex")
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, na_rep="N/A", float_format="%.1f")
    print(f"  Saved: {csv_path}")
    return [csv_path, tex_path]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    subjects = config.SUBJECTS
    training_path = config.TRAINING_GAMELOGS_PATH
    scan_path = config.GAMELOGS_PATH

    print("Extracting session timing...")
    df = extract_session_timing(subjects, training_path, scan_path)

    print("\nSession timing summary:")
    print(df.to_string(index=False))

    save_session_timing_table(df, config.TABLE_PATH)
    print(f"\nTable saved to: {config.TABLE_PATH}")
