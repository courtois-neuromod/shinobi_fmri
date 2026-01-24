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
import json
from glob import glob
from typing import Dict, List, Tuple
import logging

import shinobi_fmri.config as config
from shinobi_fmri.utils.logger import AnalysisLogger


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
) -> List[Dict]:
    """
    Extract summary statistics for a single run, potentially splitting into multiple levels.

    Args:
        sub: Subject ID (e.g., 'sub-01')
        ses: Session ID (e.g., 'ses-001')
        run: Run number (e.g., '1')
        data_path: Root data directory path

    Returns:
        List of dictionaries with run/level summary statistics
    """
    run_padded = f"{int(run):02d}"

    # Paths
    fmri_path = op.join(
        data_path, "shinobi.fmriprep", sub, ses, "func",
        f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    # Use simpler events file for level splitting
    events_path = op.join(
        data_path, "shinobi", sub, ses, "func",
        f"{sub}_{ses}_task-shinobi_run-{run_padded}_events.tsv"
    )
    # Annotated events for detailed event counts
    annotated_events_path = op.join(
        data_path, "shinobi", sub, ses, "func",
        f"{sub}_{ses}_task-shinobi_run-{run_padded}_desc-annotated_events.tsv"
    )
    
    # Check which events file exists (prefer annotated)
    if op.exists(annotated_events_path):
        events_path_to_use = annotated_events_path
        events_available = True
    elif op.exists(events_path):
        events_path_to_use = events_path
        events_available = True
    else:
        events_path_to_use = None
        events_available = False

    base_summary = {
        'subject': sub,
        'session': ses,
        'run': run_padded,
        'fmri_exists': op.exists(fmri_path),
        'events_exists': events_available,
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
            base_summary['n_volumes'] = img.shape[-1]
        except Exception as e:
            print(f"Warning: Could not load {fmri_path}: {e}")

    # Count detailed events from events file (aggregated per run)
    if events_available:
        try:
            events = pd.read_csv(events_path_to_use, sep='\t')
            event_counts = count_events_by_type(events)

            # Extract counts for specific conditions
            for condition in ['HIT', 'JUMP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'Kill', 'HealthLoss']:
                # Look for exact match or level-prefixed versions
                count = event_counts.get(condition, 0)
                # Also check for level-prefixed versions (e.g., '1-0_HIT', '4-1_HIT', '5-0_HIT')
                for key in event_counts.keys():
                    if key.endswith(f'_{condition}') or key.endswith(f'-{condition}'):
                        count += event_counts[key]
                base_summary[f'n_{condition}'] = count

        except Exception as e:
            print(f"Warning: Could not process events {annotated_events_path}: {e}")

    # Split into levels based on simple events file
    summaries = []
    
    if op.exists(events_path):
        try:
            level_events = pd.read_csv(events_path, sep='\t')
            
            # Iterate over each played level in this run
            for _, row in level_events.iterrows():
                level_summary = base_summary.copy()
                
                # Extract level info
                raw_level = str(row['level'])
                # Map 1-0 -> Level 1, 4-1 -> Level 4, etc.
                if raw_level.startswith('1-'):
                    level_name = "Level 1"
                elif raw_level.startswith('4-'):
                    level_name = "Level 4"
                elif raw_level.startswith('5-'):
                    level_name = "Level 5"
                else:
                    level_name = raw_level # Fallback
                
                level_summary['level'] = level_name
                level_summary['level_raw'] = raw_level
                level_summary['duration'] = row['duration']
                
                # Determine Cleared status from JSON sidecar
                stim_file = row.get('stim_file')
                level_summary['cleared'] = False # Default
                
                if stim_file and isinstance(stim_file, str) and stim_file != "Missing file":
                    # Path is relative to shinobi root
                    json_path = op.join(data_path, "shinobi", stim_file.replace(".bk2", "_summary.json"))
                    if op.exists(json_path):
                        try:
                            with open(json_path) as jf:
                                data = json.load(jf)
                                level_summary['cleared'] = bool(data.get('cleared', False))
                        except Exception as e:
                            # logging handled by caller/verbose check usually
                            pass
                            
                summaries.append(level_summary)
                
        except Exception as e:
            print(f"Warning: Could not process level events {events_path}: {e}")
            # Fallback if processing fails: add base summary with empty level info
            summaries.append(base_summary)
    else:
        # If no events file, just return base summary
        summaries.append(base_summary)
        
    return summaries


def generate_dataset_summary(
    data_path: str,
    output_csv: str = None,
    verbose: bool = False,
    logger: AnalysisLogger = None
) -> pd.DataFrame:
    """
    Generate complete dataset summary table.

    Args:
        data_path: Root data directory
        output_csv: Optional path to save CSV output
        verbose: Print progress information (deprecated, use logger instead)
        logger: AnalysisLogger instance for logging

    Returns:
        DataFrame with one row per run containing summary statistics
    """
    if logger:
        logger.info(f"Scanning dataset in: {data_path}")
    elif verbose:
        print(f"Scanning dataset in: {data_path}")

    # Find all fMRI files
    fmri_pattern = op.join(
        data_path, "shinobi.fmriprep", "sub-*", "ses-*", "func",
        "*_task-shinobi_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    fmri_files = sorted(glob(fmri_pattern))

    if logger:
        logger.info(f"Found {len(fmri_files)} fMRI runs")
    elif verbose:
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

        if logger:
            logger.debug(f"Processing {sub} {ses} run-{run}")
        elif verbose:
            print(f"Processing {sub} {ses} run-{run}")

        # get_run_summary now returns a list of dicts (one per level attempt)
        run_summaries = get_run_summary(sub, ses, run, data_path)
        summaries.extend(run_summaries)

    # Create DataFrame
    df = pd.DataFrame(summaries)

    # Sort by subject, session, run
    df = df.sort_values(['subject', 'session', 'run']).reset_index(drop=True)

    # Add summary statistics
    if logger or verbose:
        summary_lines = [
            "=" * 60,
            "DATASET SUMMARY",
            "=" * 60,
            f"Total level attempts: {len(df)}",
            f"Subjects: {df['subject'].nunique()}",
            f"Sessions: {df['session'].nunique()}",
            f"Runs with fMRI: {df['fmri_exists'].sum()}", # This is now duplicated per level, so it counts level attempts with fMRI
            f"Runs with events: {df['events_exists'].sum()}",
        ]
        
        if 'n_volumes' in df.columns:
             # Dedup for volume stats
             unique_runs = df.drop_duplicates(subset=['subject', 'session', 'run'])
             summary_lines.append(f"Mean volumes per run: {unique_runs['n_volumes'].mean():.1f} ± {unique_runs['n_volumes'].std():.1f}")

        summary_lines.append("")
        summary_lines.append("Event counts across dataset:")

        # Event statistics
        event_cols = [c for c in df.columns if c.startswith('n_') and c not in ['n_volumes']]
        # Event counts are also duplicated per level attempt because they come from the run-level annotated file
        # We need to take them from unique runs only to avoid overcounting
        unique_runs_df = df.drop_duplicates(subset=['subject', 'session', 'run'])
        
        for col in event_cols:
            total = unique_runs_df[col].sum()
            mean_per_run = unique_runs_df[col].mean()
            summary_lines.append(f"  {col[2:]}: {total} total ({mean_per_run:.1f} per run)")
        summary_lines.append("=" * 60)

        if logger:
            for line in summary_lines:
                if line:
                    logger.info(line)
        else:
            print("\n" + "\n".join(summary_lines))

    # Save to CSV if requested
    if output_csv:
        os.makedirs(op.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        if logger:
            logger.info(f"Saved to: {output_csv}")
        elif verbose:
            print(f"\nSaved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Shinobi dataset summary table")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV file path (default: {DATA_PATH}/processed/descriptive/dataset_summary.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (0=WARNING, 1=INFO, 2=DEBUG). Can be repeated: -v, -vv"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Custom directory for log files (default: ./logs/)"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path from config (default: use config.DATA_PATH)"
    )

    args = parser.parse_args()

    # Use data path from args or config
    data_path = args.data_path if args.data_path else config.DATA_PATH

    # Set default output path
    if args.output is None:
        args.output = op.join(data_path, "processed", "descriptive", "dataset_summary.csv")

    # Setup logger
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = AnalysisLogger(
        log_name="dataset_summary",
        verbosity=verbosity_map.get(args.verbose, logging.DEBUG),
        log_dir=args.log_dir
    )

    # Generate summary
    df = generate_dataset_summary(
        data_path=data_path,
        output_csv=args.output,
        verbose=False,  # Use logger instead
        logger=logger
    )

    # Brief summary if not verbose
    if args.verbose == 0:
        print(f"Generated summary for {len(df)} runs")
        print(f"Saved to: {args.output}")

    logger.close()


if __name__ == "__main__":
    main()
