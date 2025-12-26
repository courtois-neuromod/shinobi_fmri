#!/usr/bin/env python3
"""
Generate Dataset Summary Table

Creates a comprehensive summary table of the Shinobi fMRI dataset including:
- Number of subjects, sessions, runs per subject
- fMRI volumes per run
- Event statistics (conditions, trials)
- Data availability and quality metrics

Usage:
    python dataset_summary.py [-o output.csv] [-v]
"""
import os
import os.path as op
import argparse
import pandas as pd
import nibabel as nib
from glob import glob
from typing import Dict, List, Tuple
import logging

import shinobi_fmri.config as config


def count_events_by_type(events_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count occurrences of each event type in events DataFrame.

    Args:
        events_df: Events DataFrame with 'trial_type' column

    Returns:
        Dictionary mapping event type to count
    """
    event_counts = events_df['trial_type'].value_counts().to_dict()
    return event_counts


def get_run_summary(
    sub: str,
    ses: str,
    run: str,
    data_path: str
) -> Dict:
    """
    Extract summary statistics for a single run.

    Args:
        sub: Subject ID (e.g., 'sub-01')
        ses: Session ID (e.g., 'ses-001')
        run: Run number (e.g., '1')
        data_path: Root data directory path

    Returns:
        Dictionary with run summary statistics
    """
    run_padded = f"{int(run):02d}"

    # Paths
    fmri_path = op.join(
        data_path, "shinobi.fmriprep", sub, ses, "func",
        f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    events_path = op.join(
        data_path, "shinobi_released", "shinobi", sub, ses, "func",
        f"{sub}_{ses}_task-shinobi_run-{run_padded}_desc-annotated_events.tsv"
    )

    summary = {
        'subject': sub,
        'session': ses,
        'run': run_padded,
        'fmri_exists': op.exists(fmri_path),
        'events_exists': op.exists(events_path),
        'n_volumes': None,
        'n_HIT': 0,
        'n_JUMP': 0,
        'n_LEFT': 0,
        'n_RIGHT': 0,
        'n_UP': 0,
        'n_DOWN': 0,
        'n_Kill': 0,
        'n_HealthLoss': 0,
    }

    # Get fMRI volumes
    if op.exists(fmri_path):
        try:
            img = nib.load(fmri_path)
            summary['n_volumes'] = img.shape[-1]
        except Exception as e:
            print(f"Warning: Could not load {fmri_path}: {e}")

    # Count events
    if op.exists(events_path):
        try:
            events = pd.read_csv(events_path, sep='\t')
            event_counts = count_events_by_type(events)

            # Extract counts for specific conditions
            for condition in ['HIT', 'JUMP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'Kill', 'HealthLoss']:
                # Look for exact match or level-prefixed versions
                count = event_counts.get(condition, 0)
                # Also check for level-prefixed versions (e.g., '1-0_HIT', '4-1_HIT', '5-0_HIT')
                for key in event_counts.keys():
                    if key.endswith(f'_{condition}') or key.endswith(f'-{condition}'):
                        count += event_counts[key]
                summary[f'n_{condition}'] = count

        except Exception as e:
            print(f"Warning: Could not process events {events_path}: {e}")

    return summary


def generate_dataset_summary(data_path: str, output_csv: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Generate complete dataset summary table.

    Args:
        data_path: Root data directory
        output_csv: Optional path to save CSV output
        verbose: Print progress information

    Returns:
        DataFrame with one row per run containing summary statistics
    """
    if verbose:
        print(f"Scanning dataset in: {data_path}")

    # Find all fMRI files
    fmri_pattern = op.join(
        data_path, "shinobi.fmriprep", "sub-*", "ses-*", "func",
        "*_task-shinobi_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    fmri_files = sorted(glob(fmri_pattern))

    if verbose:
        print(f"Found {len(fmri_files)} fMRI runs")

    summaries = []

    for fmri_file in fmri_files:
        # Parse filename to extract subject, session, run
        basename = op.basename(fmri_file)
        parts = basename.split('_')
        sub = parts[0]  # sub-XX
        ses = parts[1]  # ses-XXX
        run_part = [p for p in parts if p.startswith('run-')][0]
        run = run_part.split('-')[1]  # Extract run number

        if verbose:
            print(f"Processing {sub} {ses} run-{run}")

        summary = get_run_summary(sub, ses, run, data_path)
        summaries.append(summary)

    # Create DataFrame
    df = pd.DataFrame(summaries)

    # Sort by subject, session, run
    df = df.sort_values(['subject', 'session', 'run']).reset_index(drop=True)

    # Add summary statistics
    if verbose:
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total runs: {len(df)}")
        print(f"Subjects: {df['subject'].nunique()}")
        print(f"Sessions: {df['session'].nunique()}")
        print(f"Runs with fMRI: {df['fmri_exists'].sum()}")
        print(f"Runs with events: {df['events_exists'].sum()}")
        print(f"Mean volumes per run: {df['n_volumes'].mean():.1f} ± {df['n_volumes'].std():.1f}")

        # Event statistics
        event_cols = [c for c in df.columns if c.startswith('n_') and c not in ['n_volumes']]
        print("\nEvent counts across dataset:")
        for col in event_cols:
            total = df[col].sum()
            mean_per_run = df[col].mean()
            print(f"  {col[2:]}: {total} total ({mean_per_run:.1f} per run)")
        print("="*60)

    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Shinobi dataset summary table")
    parser.add_argument(
        "-o", "--output",
        default="dataset_summary.csv",
        help="Output CSV file path (default: dataset_summary.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress and summary statistics"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path from config (default: use config.DATA_PATH)"
    )

    args = parser.parse_args()

    # Use data path from args or config
    data_path = args.data_path if args.data_path else config.DATA_PATH

    # Generate summary
    df = generate_dataset_summary(
        data_path=data_path,
        output_csv=args.output,
        verbose=args.verbose
    )

    if not args.verbose:
        print(f"Generated summary for {len(df)} runs")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
