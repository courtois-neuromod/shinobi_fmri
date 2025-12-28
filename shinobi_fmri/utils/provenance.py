"""
Provenance tracking utilities for reproducible science.

This module provides functions to capture and save metadata about
analysis computations, including git commit hashes, software versions,
parameters, and timestamps.
"""

import os
import os.path as op
import json
import subprocess
import sys
import platform
from datetime import datetime
from typing import Dict, Any, Optional, List
import importlib.metadata


def get_git_commit_hash(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Get the current git commit hash.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        Git commit hash (short form, 7 chars) or None if not a git repo
        or git is not available.
    """
    try:
        if repo_path is None:
            repo_path = os.getcwd()

        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None


def get_git_status(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get git repository status information.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        Dictionary with git status information including:
        - commit_hash: Current commit (short)
        - commit_hash_long: Current commit (full)
        - branch: Current branch name
        - uncommitted_changes: True if there are uncommitted changes
        - untracked_files: True if there are untracked files
    """
    if repo_path is None:
        repo_path = os.getcwd()

    status = {
        'commit_hash': None,
        'commit_hash_long': None,
        'branch': None,
        'uncommitted_changes': False,
        'untracked_files': False,
    }

    try:
        # Get short hash
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            status['commit_hash'] = result.stdout.strip()

        # Get long hash
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            status['commit_hash_long'] = result.stdout.strip()

        # Get branch
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            status['branch'] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ['git', '-C', repo_path, 'diff', '--quiet'],
            timeout=5
        )
        status['uncommitted_changes'] = (result.returncode != 0)

        # Check for untracked files
        result = subprocess.run(
            ['git', '-C', repo_path, 'ls-files', '--others', '--exclude-standard'],
            capture_output=True, text=True, timeout=5
        )
        status['untracked_files'] = bool(result.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return status


def get_package_version(package_name: str) -> Optional[str]:
    """
    Get version of an installed package.

    Args:
        package_name: Name of the package (e.g., 'nilearn', 'numpy')

    Returns:
        Version string or None if package not found
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_software_versions(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get versions of key software packages.

    Args:
        packages: List of package names to check. If None, uses default list
                 of common neuroimaging packages.

    Returns:
        Dictionary mapping package names to version strings
    """
    if packages is None:
        packages = [
            'python',
            'nilearn',
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'nibabel',
            'sklearn',
        ]

    versions = {}

    # Python version (special case)
    if 'python' in packages:
        versions['python'] = platform.python_version()
        packages = [p for p in packages if p != 'python']

    # Other packages
    for pkg in packages:
        # Handle sklearn -> scikit-learn mapping
        pkg_lookup = 'scikit-learn' if pkg == 'sklearn' else pkg
        version = get_package_version(pkg_lookup)
        if version:
            versions[pkg] = version

    return versions


def create_metadata(
    description: str,
    script_path: str,
    output_files: List[str],
    parameters: Dict[str, Any],
    subject: Optional[str] = None,
    session: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete metadata dictionary for an analysis output.

    Args:
        description: Human-readable description of what was computed
        script_path: Path to the script that generated this output
        output_files: List of output file paths generated
        parameters: Dictionary of analysis parameters used
        subject: Subject ID (e.g., 'sub-01'), if applicable
        session: Session ID (e.g., 'ses-001'), if applicable
        additional_info: Any additional metadata to include

    Returns:
        Complete metadata dictionary following the provenance schema
    """
    # Get repository root (assumes script is in shinobi_fmri/)
    script_dir = op.dirname(op.abspath(script_path))

    # Try to find repo root starting from script directory
    repo_root = script_dir
    found_git = False
    for _ in range(5):  # Search up to 5 levels
        if op.exists(op.join(repo_root, '.git')):
            found_git = True
            break
        parent = op.dirname(repo_root)
        if parent == repo_root:  # Reached filesystem root
            break
        repo_root = parent

    # If .git not found from script path, try current working directory
    # (useful when script is run from repo root, e.g., via SLURM)
    if not found_git:
        cwd = os.getcwd()
        # Search from current working directory
        repo_root = cwd
        for _ in range(5):
            if op.exists(op.join(repo_root, '.git')):
                found_git = True
                break
            parent = op.dirname(repo_root)
            if parent == repo_root:
                break
            repo_root = parent

    git_status = get_git_status(repo_root if found_git else None)

    metadata = {
        'description': description,
        'generated_by': {
            'script_path': op.abspath(script_path),
            'script_name': op.basename(script_path),
        },
        'timestamp': datetime.now().isoformat(),
        'git': git_status,
        'subject': subject,
        'session': session,
        'parameters': parameters,
        'output_files': [op.abspath(f) for f in output_files],
        'software_versions': get_software_versions(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
        }
    }

    # Add warning if uncommitted changes
    if git_status.get('uncommitted_changes') or git_status.get('untracked_files'):
        metadata['warnings'] = [
            "Git repository has uncommitted changes or untracked files. "
            "Results may not be perfectly reproducible from the commit hash alone."
        ]

    # Add any additional info
    if additional_info:
        metadata['additional_info'] = additional_info

    return metadata


def save_metadata_json(
    metadata: Dict[str, Any],
    output_path: str,
    logger: Optional[Any] = None
) -> bool:
    """
    Save metadata to a JSON file.

    Args:
        metadata: Metadata dictionary to save
        output_path: Path where JSON file should be saved
        logger: Optional logger instance for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(op.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        if logger:
            logger.debug(f"Metadata saved to: {output_path}")

        return True

    except Exception as e:
        if logger:
            logger.error(f"Failed to save metadata to {output_path}: {e}")
        return False


def save_sidecar_metadata(
    nifti_path: str,
    metadata: Dict[str, Any],
    logger: Optional[Any] = None
) -> Optional[str]:
    """
    Save metadata as a JSON sidecar file next to a NIfTI file.

    For a NIfTI file at /path/to/file.nii.gz, this creates
    /path/to/file.json with the metadata.

    Args:
        nifti_path: Path to the NIfTI file
        metadata: Metadata dictionary to save
        logger: Optional logger instance

    Returns:
        Path to the created JSON file, or None if failed
    """
    # Remove .nii.gz or .nii extension and add .json
    if nifti_path.endswith('.nii.gz'):
        json_path = nifti_path[:-7] + '.json'
    elif nifti_path.endswith('.nii'):
        json_path = nifti_path[:-4] + '.json'
    else:
        json_path = nifti_path + '.json'

    success = save_metadata_json(metadata, json_path, logger=logger)
    return json_path if success else None


def load_metadata_json(json_path: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a JSON file.

    Args:
        json_path: Path to JSON metadata file

    Returns:
        Metadata dictionary or None if failed
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def create_dataset_description(
    name: str,
    description: str,
    pipeline_version: str,
    derived_from: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a BIDS-compatible dataset_description.json for processed data.

    Args:
        name: Name of the pipeline/analysis (e.g., 'run-level GLM')
        description: Longer description of what this pipeline does
        pipeline_version: Version string for the pipeline
        derived_from: Description of source data
        parameters: Key parameters used in this pipeline
        output_dir: If provided, save dataset_description.json to this directory

    Returns:
        Dataset description dictionary
    """
    git_status = get_git_status()

    dataset_desc = {
        'Name': name,
        'BIDSVersion': '1.9.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'shinobi_fmri',
                'Version': pipeline_version,
                'CodeURL': 'https://github.com/courtois-neuromod/shinobi_fmri',
                'GitCommit': git_status.get('commit_hash_long'),
                'GitBranch': git_status.get('branch'),
            }
        ],
        'Description': description,
    }

    if derived_from:
        dataset_desc['SourceDatasets'] = [{'Description': derived_from}]

    if parameters:
        dataset_desc['PipelineParameters'] = parameters

    # Add timestamp
    dataset_desc['DateGenerated'] = datetime.now().isoformat()

    # Save if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = op.join(output_dir, 'dataset_description.json')
        with open(output_path, 'w') as f:
            json.dump(dataset_desc, f, indent=2, default=str)

    return dataset_desc
