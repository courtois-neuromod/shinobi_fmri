"""
Validation utilities for checking analysis outputs.

Provides reusable functions for:
- File existence checking
- NIfTI file integrity validation
- Pickle file validation
- Subject/session/run discovery from raw BIDS data
- Correlation matrix validation
"""

from typing import List, Tuple, Optional, Dict, Set
import os
import os.path as op
import re
import pickle
import numpy as np


class ValidationResult:
    """Container for validation results."""

    def __init__(self, category: str):
        """
        Initialize validation result.

        Args:
            category: Name of the analysis category being validated
        """
        self.category = category
        self.expected = 0
        self.found = 0
        self.missing = []
        self.errors = []

    def add_expected(self, filepath: str):
        """
        Add an expected file to the validation.

        Args:
            filepath: Path to the expected file
        """
        self.expected += 1
        if op.exists(filepath):
            self.found += 1
        else:
            self.missing.append(filepath)

    def add_error(self, filepath: str, error: str):
        """
        Add an error encountered during validation.

        Args:
            filepath: Path to the file with error
            error: Error message
        """
        self.errors.append((filepath, error))

    @property
    def completion_rate(self) -> float:
        """
        Calculate completion rate as percentage.

        Returns:
            Percentage of expected files that were found
        """
        if self.expected == 0:
            return 0.0
        return (self.found / self.expected) * 100


def check_file_exists(filepath: str) -> bool:
    """
    Check if file exists.

    Args:
        filepath: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return op.exists(filepath)


def validate_nifti(filepath: str, check_integrity: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate NIfTI file.

    Args:
        filepath: Path to .nii.gz file
        check_integrity: If True, load and check dimensions

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not op.exists(filepath):
        return False, "File does not exist"

    if not check_integrity:
        return True, None

    try:
        import nibabel as nib
        img = nib.load(filepath)
        if img.shape is None or len(img.shape) == 0:
            return False, "Invalid image dimensions"
        return True, None
    except Exception as e:
        return False, f"Failed to load: {str(e)}"


def validate_pickle(filepath: str, check_integrity: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate pickle file.

    Args:
        filepath: Path to .pkl file
        check_integrity: If True, try loading the pickle

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not op.exists(filepath):
        return False, "File does not exist"

    if not check_integrity:
        return True, None

    try:
        import pickle
        with open(filepath, 'rb') as f:
            _ = pickle.load(f)
        return True, None
    except Exception as e:
        return False, f"Failed to load: {str(e)}"


def get_available_runs(data_path: str, subject: str, session: str) -> List[str]:
    """
    Get list of available run numbers for a subject/session from raw BIDS data.

    Args:
        data_path: Root data directory
        subject: Subject ID (e.g., 'sub-01')
        session: Session ID (e.g., 'ses-001')

    Returns:
        List of run numbers as strings (e.g., ['01', '02', '03'])
    """
    # First try raw BIDS data
    func_dir = op.join(data_path, "shinobi", subject, session, "func")

    # Fallback to fmriprep if raw data not found
    if not op.exists(func_dir):
        func_dir = op.join(data_path, "shinobi.fmriprep", subject, session, "func")

    if not op.exists(func_dir):
        return []

    files = os.listdir(func_dir)
    # Look for BOLD files (either raw or preprocessed)
    run_files = [f for f in files if ("_bold.nii.gz" in f or "desc-preproc_bold.nii.gz" in f)]

    runs = []
    for fname in run_files:
        match = re.search(r'run-(\d+)', fname)
        if match:
            runs.append(match.group(1))

    return sorted(set(runs))  # Remove duplicates and sort


def get_subjects_from_raw_data(data_path: str) -> List[str]:
    """
    Get list of subjects from raw BIDS data.

    Args:
        data_path: Root data directory

    Returns:
        List of subject IDs (e.g., ['sub-01', 'sub-02'])
    """
    bids_dir = op.join(data_path, "shinobi")
    if not op.exists(bids_dir):
        return []

    subjects = [d for d in os.listdir(bids_dir)
                if d.startswith('sub-') and op.isdir(op.join(bids_dir, d))]
    return sorted(subjects)


def get_sessions_from_raw_data(data_path: str, subject: str) -> List[str]:
    """
    Get list of sessions for a subject from raw BIDS data.

    Args:
        data_path: Root data directory
        subject: Subject ID (e.g., 'sub-01')

    Returns:
        List of session IDs (e.g., ['ses-001', 'ses-002'])
    """
    subject_dir = op.join(data_path, "shinobi", subject)
    if not op.exists(subject_dir):
        return []

    sessions = [d for d in os.listdir(subject_dir)
                if d.startswith('ses-') and op.isdir(op.join(subject_dir, d))]
    return sorted(sessions)


def load_correlation_matrix(filepath: str) -> Optional[Dict]:
    """
    Load correlation matrix pickle file.

    Args:
        filepath: Path to correlation matrix pickle file

    Returns:
        Dictionary containing correlation matrix data, or None if load fails
    """
    if not op.exists(filepath):
        return None

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        return None


def validate_correlation_matrix(corr_data: Dict, subjects: List[str],
                                conditions: List[str]) -> Tuple[Dict, List[str]]:
    """
    Validate correlation matrix completeness.

    Checks which subject/session/condition combinations are present in the
    correlation matrix and reports which are missing.

    Args:
        corr_data: Correlation matrix dictionary (loaded from pickle)
        subjects: List of subjects to check for
        conditions: List of conditions to check for

    Returns:
        Tuple of (stats_dict, missing_entries):
            - stats_dict: Statistics about matrix completeness
            - missing_entries: List of missing subject/session/condition combinations
    """
    if corr_data is None:
        return {'total_maps': 0, 'expected_maps': 0, 'missing_maps': 0}, []

    # Extract map information
    map_subjects = corr_data.get('subj', [])
    map_sessions = corr_data.get('ses', [])
    map_conditions = corr_data.get('cond', [])
    map_sources = corr_data.get('source', [])
    corr_matrix = corr_data.get('corr_matrix', None)
    computed_mask = corr_data.get('computed_mask', None)

    # Build set of existing maps
    existing_maps = set()
    for i in range(len(map_subjects)):
        subj = map_subjects[i]
        ses = map_sessions[i]
        cond = map_conditions[i]
        source = map_sources[i]

        # Only count shinobi session-level maps (not HCP)
        if source == 'session-level' and subj in subjects:
            existing_maps.add((subj, ses, cond))

    # Determine expected maps
    # We need to scan the actual data to know which sessions exist
    # For now, we'll just report what's in the matrix

    # Check correlation matrix completeness
    missing_correlations = []
    if corr_matrix is not None and computed_mask is not None:
        n_maps = corr_matrix.shape[0]
        total_pairs = n_maps * (n_maps - 1) // 2  # Upper triangle

        # Count computed correlations
        if computed_mask.shape == corr_matrix.shape:
            computed_pairs = np.sum(np.triu(computed_mask, k=1))
        else:
            computed_pairs = 0

        missing_pair_count = total_pairs - computed_pairs
    else:
        total_pairs = 0
        computed_pairs = 0
        missing_pair_count = 0

    stats = {
        'total_maps_in_matrix': len(map_subjects),
        'shinobi_session_maps': len(existing_maps),
        'total_possible_pairs': total_pairs,
        'computed_pairs': int(computed_pairs),
        'missing_pairs': int(missing_pair_count),
        'matrix_completion_rate': (computed_pairs / total_pairs * 100) if total_pairs > 0 else 0
    }

    # Report which subject/session/condition combos are present
    missing_maps = []
    for subj in subjects:
        for cond in conditions:
            # Check if this subject+condition has ANY sessions in matrix
            has_any_session = any((s, se, c) in existing_maps
                                 for s, se, c in existing_maps
                                 if s == subj and c == cond)
            if not has_any_session:
                missing_maps.append(f"{subj} condition {cond}")

    return stats, missing_maps


def get_expected_beta_maps(data_path: str, subjects: List[str],
                          conditions: List[str]) -> Set[Tuple[str, str, str]]:
    """
    Get expected beta maps based on available raw data.

    Scans raw BIDS data to determine which subject/session/condition
    combinations should have beta maps.

    Args:
        data_path: Root data directory
        subjects: List of subjects
        conditions: List of conditions

    Returns:
        Set of (subject, session, condition) tuples
    """
    expected = set()

    for subject in subjects:
        sessions = get_sessions_from_raw_data(data_path, subject)
        for session in sessions:
            runs = get_available_runs(data_path, subject, session)
            if runs:  # Only expect session-level maps if there are runs
                for condition in conditions:
                    expected.add((subject, session, condition))

    return expected
