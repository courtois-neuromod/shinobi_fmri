#!/usr/bin/env python
"""
Diagnose missing z-maps for GLM analysis.

This script checks which z-maps are missing and investigates why they weren't created:
- Missing input files (fMRI, events, masks)
- Failed GLM computations
- Insufficient events for a condition

Usage:
    python diagnose_missing_zmaps.py --level session
    python diagnose_missing_zmaps.py --level subject
    python diagnose_missing_zmaps.py --subject sub-01 --session ses-005
"""

import os
import os.path as op
import argparse
import pandas as pd
from typing import List, Dict, Tuple

try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS
except ImportError:
    DATA_PATH = "/home/hyruuk/scratch/data"
    SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']

CONDITIONS = ['HIT', 'JUMP', 'DOWN', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP']


def check_fmri_files(subject: str, session: str) -> Tuple[bool, List[str], str]:
    """
    Check if fMRI files exist for a session.

    Returns:
        (all_exist, run_list, error_msg)
    """
    fmriprep_dir = op.join(DATA_PATH, "shinobi.fmriprep", subject, session, "func")

    if not op.exists(fmriprep_dir):
        return False, [], f"fMRIPrep directory not found: {fmriprep_dir}"

    files = os.listdir(fmriprep_dir)
    run_files = [
        f for f in files
        if "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in f
    ]

    if not run_files:
        return False, [], f"No preprocessed fMRI files found in {fmriprep_dir}"

    run_list = sorted([f.split('run-')[1][0] for f in run_files])
    return True, run_list, ""


def check_events_files(subject: str, session: str, run_list: List[str]) -> Tuple[bool, List[str], str]:
    """
    Check if annotated events files exist for all runs.

    Returns:
        (all_exist, missing_runs, error_msg)
    """
    missing_runs = []

    for run in run_list:
        events_file = op.join(
            DATA_PATH, "shinobi", subject, session, "func",
            f"{subject}_{session}_task-shinobi_run-0{run}_desc-annotated_events.tsv"
        )

        if not op.exists(events_file):
            missing_runs.append(run)

    if missing_runs:
        return False, missing_runs, f"Missing annotated events for runs: {missing_runs}"

    return True, [], ""


def check_condition_events(subject: str, session: str, condition: str, run_list: List[str]) -> Tuple[bool, Dict, str]:
    """
    Check if a condition has events in the session.

    Returns:
        (has_events, events_per_run, error_msg)
    """
    events_per_run = {}
    total_events = 0

    for run in run_list:
        events_file = op.join(
            DATA_PATH, "shinobi", subject, session, "func",
            f"{subject}_{session}_task-shinobi_run-0{run}_desc-annotated_events.tsv"
        )

        if not op.exists(events_file):
            continue

        try:
            events_df = pd.read_csv(events_file, sep='\t')
            condition_events = events_df[events_df['trial_type'] == condition]
            n_events = len(condition_events)
            events_per_run[run] = n_events
            total_events += n_events
        except Exception as e:
            events_per_run[run] = f"ERROR: {e}"

    if total_events == 0:
        return False, events_per_run, f"No '{condition}' events found in any run"

    return True, events_per_run, ""


def check_zmap_exists(subject: str, session: str, condition: str, level: str = "ses-level") -> bool:
    """Check if z-map file exists."""
    zmap_path = op.join(
        DATA_PATH, "processed", "z_maps", level, condition,
        f"{subject}_{session}_{condition}.nii.gz"
    )
    return op.exists(zmap_path)


def check_glm_exists(subject: str, session: str, condition: str, level: str = "ses-level") -> bool:
    """Check if GLM pickle file exists."""
    glm_path = op.join(
        DATA_PATH, "processed", "glm", level,
        f"{subject}_{session}_{condition}_fitted_glm.pkl"
    )
    return op.exists(glm_path)


def diagnose_session(subject: str, session: str, condition: str) -> Dict:
    """
    Diagnose why a z-map might be missing for a session.

    Returns:
        Dictionary with diagnostic information
    """
    result = {
        'subject': subject,
        'session': session,
        'condition': condition,
        'zmap_exists': False,
        'glm_exists': False,
        'fmri_ok': False,
        'events_ok': False,
        'condition_events_ok': False,
        'run_list': [],
        'issues': []
    }

    # Check z-map
    result['zmap_exists'] = check_zmap_exists(subject, session, condition)
    result['glm_exists'] = check_glm_exists(subject, session, condition)

    # Check fMRI files
    fmri_ok, run_list, fmri_error = check_fmri_files(subject, session)
    result['fmri_ok'] = fmri_ok
    result['run_list'] = run_list

    if not fmri_ok:
        result['issues'].append(f"FMRI: {fmri_error}")
        return result

    # Check events files
    events_ok, missing_runs, events_error = check_events_files(subject, session, run_list)
    result['events_ok'] = events_ok

    if not events_ok:
        result['issues'].append(f"EVENTS: {events_error}")

    # Check condition-specific events
    cond_ok, events_per_run, cond_error = check_condition_events(subject, session, condition, run_list)
    result['condition_events_ok'] = cond_ok
    result['events_per_run'] = events_per_run

    if not cond_ok:
        result['issues'].append(f"CONDITION: {cond_error}")

    # If everything is OK but z-map is missing, it's a computation failure
    if fmri_ok and events_ok and cond_ok and not result['zmap_exists']:
        if result['glm_exists']:
            result['issues'].append("GLM exists but z-map not computed - computation may have failed during contrast computation")
        else:
            result['issues'].append("All inputs exist but GLM/z-map not computed - processing may not have been run")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--level', type=str, default='session', choices=['session', 'subject'],
                       help='Analysis level to check')
    parser.add_argument('--subject', type=str, default=None,
                       help='Specific subject to check (default: all)')
    parser.add_argument('--session', type=str, default=None,
                       help='Specific session to check (requires --subject)')
    parser.add_argument('--condition', type=str, default=None,
                       help='Specific condition to check (default: all)')
    parser.add_argument('--missing-only', action='store_true',
                       help='Only show missing z-maps')

    args = parser.parse_args()

    if args.session and not args.subject:
        parser.error("--session requires --subject")

    # Determine subjects to check
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = SUBJECTS

    # Determine conditions to check
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = CONDITIONS

    print(f"\n{'='*80}")
    print(f"Diagnosing {args.level}-level z-maps")
    print(f"{'='*80}\n")

    all_diagnostics = []

    for subject in subjects:
        # Get sessions for this subject
        shinobi_dir = op.join(DATA_PATH, "shinobi", subject)
        if not op.exists(shinobi_dir):
            print(f"⚠ Warning: Subject directory not found: {shinobi_dir}")
            continue

        if args.session:
            sessions = [args.session]
        else:
            sessions = sorted([d for d in os.listdir(shinobi_dir) if d.startswith('ses-')])

        for session in sessions:
            for condition in conditions:
                diag = diagnose_session(subject, session, condition)
                all_diagnostics.append(diag)

                # Print results
                if args.missing_only and diag['zmap_exists']:
                    continue

                status = "✓ EXISTS" if diag['zmap_exists'] else "✗ MISSING"
                print(f"\n{subject} {session} {condition}: {status}")

                if not diag['zmap_exists']:
                    print(f"  Runs: {', '.join(diag['run_list']) if diag['run_list'] else 'NONE'}")

                    if diag['issues']:
                        print(f"  Issues:")
                        for issue in diag['issues']:
                            print(f"    - {issue}")

                    if diag.get('events_per_run'):
                        print(f"  Events per run:")
                        for run, n_events in diag['events_per_run'].items():
                            print(f"    run-{run}: {n_events}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total = len(all_diagnostics)
    missing = sum(1 for d in all_diagnostics if not d['zmap_exists'])
    existing = total - missing

    print(f"Total checked: {total}")
    print(f"Existing:      {existing}")
    print(f"Missing:       {missing}")

    if missing > 0:
        print(f"\nMissing breakdown by issue:")

        issue_counts = {}
        for diag in all_diagnostics:
            if not diag['zmap_exists']:
                for issue in diag['issues']:
                    issue_type = issue.split(':')[0]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type}: {count}")

    print()


if __name__ == "__main__":
    main()
