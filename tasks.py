"""
Shinobi fMRI Analysis Pipeline - Task Automation with Airoh/Invoke

This file provides a unified interface for running the shinobi_fmri analysis pipeline,
supporting both local execution and SLURM-based cluster computing.

Usage:
    invoke --list                                               # Show all available tasks
    invoke glm.run-level --subject sub-01 --session ses-001     # Run single subject/session locally
    invoke glm.run-level --subject sub-01 --session ses-001 --slurm  # Submit single job to SLURM
    invoke glm.run-level                                        # Run all subjects/sessions locally
    invoke glm.run-level --slurm                                # Submit all to SLURM (batch mode)
    invoke glm.run-level --n-jobs 8                             # Use 8 CPU cores (default: -1 = all)
"""

from invoke import task, Collection
import os
import os.path as op
import glob

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import airoh utilities (optional)
try:
    from airoh import utils, containers
    AIROH_AVAILABLE = True
except ImportError:
    AIROH_AVAILABLE = False
    print("Note: airoh not installed. Install with 'pip install airoh invoke' for additional utilities.")

# Import configuration
try:
    from shinobi_fmri.config import DATA_PATH, SUBJECTS, CONDITIONS, PYTHON_BIN, SLURM_PYTHON_BIN
except ImportError:
    # Fallback to environment variables and defaults
    DATA_PATH = os.getenv("SHINOBI_DATA_PATH", "/home/hyruuk/scratch/data")
    SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
    CONDITIONS = ['HIT', 'JUMP', 'DOWN', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP']
    PYTHON_BIN = os.getenv("SHINOBI_PYTHON_BIN", "python")
    SLURM_PYTHON_BIN = os.getenv("SHINOBI_SLURM_PYTHON_BIN", "python")

    if "SHINOBI_DATA_PATH" not in os.environ and not op.exists(DATA_PATH):
        print(f"Warning: Default DATA_PATH {DATA_PATH} does not exist. Set SHINOBI_DATA_PATH env var.")
    else:
        print(f"Using DATA_PATH: {DATA_PATH}")


# =============================================================================
# Configuration
# =============================================================================

# Project paths (these are computed, not from config)
PROJECT_ROOT = op.dirname(op.abspath(__file__))
SHINOBI_FMRI_DIR = op.join(PROJECT_ROOT, "shinobi_fmri")
SLURM_DIR = op.join(PROJECT_ROOT, "slurm")

# Note: DATA_PATH, SUBJECTS, CONDITIONS, PYTHON_BIN, and SLURM_PYTHON_BIN
# are now loaded from config.py above (or fall back to environment variables)


# =============================================================================
# Helper Functions
# =============================================================================

def get_subject_sessions(data_path, subject=None):
    """Get all available subject-session combinations from fMRIPrep output, in sorted order."""
    fmriprep_dir = op.join(data_path, "shinobi.fmriprep")

    if subject:
        subjects = [subject]
    else:
        subjects = sorted([d for d in os.listdir(fmriprep_dir)
                          if d.startswith('sub-') and op.isdir(op.join(fmriprep_dir, d))])

    sub_ses_pairs = []
    for sub in subjects:
        sub_dir = op.join(fmriprep_dir, sub)
        if op.exists(sub_dir):
            sessions = sorted([d for d in os.listdir(sub_dir)
                             if d.startswith('ses-') and op.isdir(op.join(sub_dir, d))])
            for ses in sessions:
                sub_ses_pairs.append((sub, ses))

    return sub_ses_pairs


def get_all_runs(data_path):
    """Get all available runs (subject-session pairs) from fMRIPrep output, in sorted order."""
    pattern = op.join(
        data_path,
        "shinobi.fmriprep",
        "*",
        "*",
        "func",
        "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    filelist = glob.glob(pattern)

    runs = []
    for file in filelist:
        basename = op.basename(file)
        parts = basename.split("_")
        sub = parts[0]
        ses = parts[1]
        runs.append((sub, ses))

    # Sort by subject, then session
    return sorted(set(runs))


# =============================================================================
# GLM Analysis Tasks
# =============================================================================

# =============================================================================
# GLM Analysis Tasks
# =============================================================================

@task
def glm_run_level(c, subject=None, session=None, slurm=False, n_jobs=-1, verbose=0, log_dir=None, low_level_confs=False):
    """
    Run run-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01) - if None, process all subjects
        session: Session ID (e.g., ses-001) - if None, process all sessions
        slurm: If True, submit to SLURM cluster (single job or batch mode depending on subject/session)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_dir: Custom log directory
        low_level_confs: If True, include low-level confounds and button-press rate in design matrix
    """
    # If no subject specified, process all subjects/sessions
    if subject is None:
        runs = get_all_runs(DATA_PATH)
        print(f"Processing {len(runs)} runs for all subjects...")
        for sub, ses in runs:
            glm_run_level(c, subject=sub, session=ses, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir, low_level_confs=low_level_confs)
        return

    # If subject specified but no session, process all sessions for that subject
    if session is None:
        sub_ses_pairs = get_subject_sessions(DATA_PATH, subject=subject)
        print(f"Processing {len(sub_ses_pairs)} sessions for {subject}...")
        for sub, ses in sub_ses_pairs:
            glm_run_level(c, subject=sub, session=ses, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir, low_level_confs=low_level_confs)
        return

    # Process single subject/session
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_run_level.py")

    args = f"--subject {subject} --session {session}"

    # Handle verbosity
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    elif isinstance(verbose, bool) and verbose:
        args += " -v"

    if log_dir:
        args += f" --log-dir {log_dir}"

    if low_level_confs:
        args += " --low-level-confs"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_run-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {session}"
        print(f"Submitting to SLURM: {subject} {session}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def glm_session_level(c, subject=None, session=None, slurm=False, n_jobs=-1, verbose=0, log_dir=None, low_level_confs=False):
    """
    Run session-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01) - if None, process all subjects
        session: Session ID (e.g., ses-001) - if None, process all sessions
        slurm: If True, submit to SLURM cluster (single job or batch mode depending on subject/session)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        verbose: Verbosity level
        log_dir: Custom log directory
        low_level_confs: If True, include low-level confounds and button-press rate in design matrix
    """
    # If no subject specified, process all subjects/sessions
    if subject is None:
        sub_ses_pairs = get_subject_sessions(DATA_PATH)
        print(f"Processing {len(sub_ses_pairs)} sessions for all subjects...")
        for sub, ses in sub_ses_pairs:
            glm_session_level(c, subject=sub, session=ses, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir, low_level_confs=low_level_confs)
        return

    # If subject specified but no session, process all sessions for that subject
    if session is None:
        sub_ses_pairs = get_subject_sessions(DATA_PATH, subject=subject)
        print(f"Processing {len(sub_ses_pairs)} sessions for {subject}...")
        for sub, ses in sub_ses_pairs:
            glm_session_level(c, subject=sub, session=ses, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir, low_level_confs=low_level_confs)
        return

    # Process single subject/session
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_session_level.py")

    args = f"--subject {subject} --session {session}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"
    if low_level_confs:
        args += " --low-level-confs"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_session-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {session}"
        print(f"Submitting to SLURM: {subject} {session}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def glm_subject_level(c, subject=None, condition=None, slurm=False, n_jobs=-1, verbose=0, log_dir=None):
    """
    Run subject-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01) - if None, process all subjects
        condition: Condition/contrast name (e.g., HIT, JUMP, Kill) - if None, process all conditions
        slurm: If True, submit to SLURM cluster (single job or batch mode depending on subject/condition)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    # If no subject specified, process all subjects
    if subject is None:
        subjects = SUBJECTS
    else:
        subjects = [subject]

    # If no condition specified, process all conditions
    if condition is None:
        conditions = CONDITIONS
    else:
        conditions = [condition]

    # If processing multiple subjects/conditions, iterate
    if len(subjects) > 1 or len(conditions) > 1:
        print(f"Processing {len(subjects)} subject(s) x {len(conditions)} condition(s) = {len(subjects)*len(conditions)} jobs...")
        for sub in subjects:
            for cond in conditions:
                glm_subject_level(c, subject=sub, condition=cond, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir)
        return

    # Process single subject/condition
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_subject_level.py")

    args = f"--subject {subjects[0]} --condition {conditions[0]}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_subject-level.sh")
        cmd = f"sbatch {slurm_script} {subjects[0]} {conditions[0]}"
        print(f"Submitting to SLURM: {subjects[0]} {conditions[0]}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


# =============================================================================
# MVPA Tasks
# =============================================================================

@task
def mvpa_session_level(c, subject, task_name=None, perm_index=0, slurm=False, n_jobs=-1, verbose=0, log_dir=None):
    """
    Run session-level MVPA analysis.

    Args:
        subject: Subject ID (e.g., sub-01)
        task_name: Task name (kept for compatibility, though compute_mvpa might not strictly need it if it processes all conditions)
        perm_index: Permutation index for significance testing
        slurm: If True, submit to SLURM cluster
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "mvpa", "compute_mvpa.py")

    # Note: compute_mvpa.py args might differ slightly (screening, etc)
    # We'll map what we have. If 'task' arg used to be essential, we preserve it if the script uses it.
    # Looking at compute_mvpa.py, it takes --subject.
    args = f"--subject {subject} --n_jobs {n_jobs}"

    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_mvpa_ses-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {task_name} {perm_index}"
        print(f"Submitting to SLURM: {subject} {task_name} perm={perm_index}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


# =============================================================================
# Correlation Analysis Tasks
# =============================================================================

@task
def correlations(c, chunk_start=0, chunk_size=100, n_jobs=-1, slurm=False, verbose=0, log_dir=None):
    """
    Compute correlation matrices with HCP data.

    Args:
        chunk_start: Starting index for chunked processing
        chunk_size: Number of maps per chunk
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        slurm: If True, submit to SLURM cluster
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "correlations", "session_corrmat_with_hcp.py")

    args = f"--chunk-start {chunk_start} --chunk-size {chunk_size} --n-jobs {n_jobs}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_corrmat_chunk.sh")
        cmd = f"sbatch {slurm_script} {chunk_start}"
        print(f"Submitting to SLURM: chunk_start={chunk_start}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


# =============================================================================
# Visualization Tasks
# =============================================================================

@task
def viz_run_level(c, subject, condition, slurm=False, verbose=0, log_dir=None):
    """
    Generate run-level visualizations.

    Args:
        subject: Subject ID (e.g., sub-01)
        condition: Condition/contrast name
        slurm: If True, submit to SLURM cluster
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_run-level.py")
    
    args = f"--subject {subject} --condition {condition}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-run-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {condition}"
        print(f"Submitting to SLURM: {subject} {condition}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def viz_session_level(c, subject=None, condition=None, slurm=False, verbose=0, log_dir=None):
    """
    Generate session-level visualizations.

    Args:
        subject: Subject ID (e.g., sub-01) - Optional, process all if None (dep on script)
        condition: Condition (e.g., HIT) - Optional
        slurm: If True, submit to SLURM cluster
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_session-level.py")
    
    args = ""
    if subject:
        args += f" --subject {subject}"
    if condition:
        args += f" --contrast {condition}"
        
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-sesslevel.sh")
        # Ensure script handling matches
        cmd = f"sbatch {slurm_script}"
        print("Submitting to SLURM")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def viz_subject_level(c, slurm=False, verbose=0, log_dir=None):
    """
    Generate subject-level visualizations.

    Args:
        slurm: If True, submit to SLURM cluster
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_subject-level.py")
    
    args = ""
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-sub-level.sh")
        cmd = f"sbatch {slurm_script}"
        print("Submitting to SLURM")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def viz_annotation_panels(c, condition=None, conditions=None, skip_individual=False, skip_panels=False, skip_pdf=False, verbose=0, log_dir=None):
    """
    Generate annotation panels with subject-level and session-level brain maps.

    Creates:
    - Individual inflated brain maps for each subject/session
    - Combined panels (1 subject-level + top 4 session-level maps per subject)
    - PDF with all annotation panels

    Args:
        condition: Single condition to process (e.g., 'HIT')
        conditions: Comma-separated conditions (e.g., 'HIT,JUMP,Kill')
        skip_individual: Skip generating individual brain maps
        skip_panels: Skip generating annotation panels
        skip_pdf: Skip generating PDF
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_annotation_panels.py")

    cmd_parts = [PYTHON_BIN, script]

    if condition:
        cmd_parts.extend(['--condition', condition])
    elif conditions:
        cmd_parts.extend(['--conditions', conditions])

    if skip_individual:
        cmd_parts.append('--skip-individual')
    if skip_panels:
        cmd_parts.append('--skip-panels')
    if skip_pdf:
        cmd_parts.append('--skip-pdf')
        
    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)
    print(f"Generating annotation panels...")
    if condition:
        print(f"  Condition: {condition}")
    elif conditions:
        print(f"  Conditions: {conditions}")
    else:
        print(f"  Processing all default conditions")

    c.run(cmd)


@task
def viz_beta_correlations(c, input_path, output_path, verbose=0, log_dir=None):
    """
    Generate beta correlations figure.

    Args:
        input_path: Path to input .pkl file (e.g. ses-level_beta_maps_ICC.pkl)
        output_path: Path to save the output figure
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "beta_correlations_plot.py")

    cmd = f"{PYTHON_BIN} {script} --input {input_path} --output {output_path}"

    print(f"Generating beta correlations figure...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    c.run(cmd)


@task
def viz_regressor_correlations(c, subject=None, skip_generation=False, low_level_confs=False, verbose=0, log_dir=None):
    """
    Generate correlation matrices for design matrix regressors.

    Creates correlation heatmaps and clustermaps showing relationships between
    all regressors (annotations) in the GLM design matrices. Produces both
    per-run and subject-averaged visualizations.

    Args:
        subject: Specific subject to process (default: all subjects)
        skip_generation: Skip design matrix generation, only plot from existing pickle
        low_level_confs: Include low-level confounds (psychophysics and button presses)
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_regressor_correlations.py")

    cmd_parts = [PYTHON_BIN, script]

    if subject:
        cmd_parts.extend(['--subject', subject])
    if skip_generation:
        cmd_parts.append('--skip-generation')
    if low_level_confs:
        cmd_parts.append('--low-level-confs')

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)

    print(f"Generating regressor correlation matrices...")
    if subject:
        print(f"  Subject: {subject}")
    else:
        print(f"  Processing all subjects")
    if low_level_confs:
        print(f"  Including low-level confounds")

    c.run(cmd)


# =============================================================================
# Diagnostic Tasks
# =============================================================================

@task
def diagnose_missing_zmaps(c, level='session', subject=None, session=None, condition=None, missing_only=False):
    """
    Diagnose why z-maps are missing for GLM analysis.

    Checks for:
    - Missing input files (fMRI, events, masks)
    - Insufficient events for a condition
    - Failed GLM computations

    Args:
        level: Analysis level to check ('session' or 'subject')
        subject: Specific subject to check (default: all)
        session: Specific session to check (requires subject)
        condition: Specific condition to check (default: all)
        missing_only: Only show missing z-maps
    """
    script = op.join(SHINOBI_FMRI_DIR, "glm", "diagnose_missing_zmaps.py")

    cmd_parts = [PYTHON_BIN, script, '--level', level]

    if subject:
        cmd_parts.extend(['--subject', subject])
    if session:
        cmd_parts.extend(['--session', session])
    if condition:
        cmd_parts.extend(['--condition', condition])
    if missing_only:
        cmd_parts.append('--missing-only')

    cmd = ' '.join(cmd_parts)
    print(f"Diagnosing {level}-level z-maps...")
    c.run(cmd)


# =============================================================================
# Full Pipeline Tasks
# =============================================================================

@task
def pipeline_full(c, subject, session, slurm=False, n_jobs=-1):
    """
    Run complete analysis pipeline for a single subject/session.

    This runs: run-level GLM -> session-level GLM -> visualizations

    Args:
        subject: Subject ID (e.g., sub-01)
        session: Session ID (e.g., ses-001)
        slurm: If True, submit each stage to SLURM
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
    """
    print(f"\n{'='*60}")
    print(f"Running full pipeline for {subject} {session}")
    print(f"{'='*60}\n")

    print("\n[1/3] Run-level GLM...")
    glm_run_level(c, subject, session, slurm=slurm, n_jobs=n_jobs)

    print("\n[2/3] Session-level GLM...")
    glm_session_level(c, subject, session, slurm=slurm, n_jobs=n_jobs)

    print("\n[3/3] Visualizations...")
    viz_session_level(c, slurm=slurm)

    print(f"\n{'='*60}")
    print(f"Pipeline complete for {subject} {session}!")
    print(f"{'='*60}\n")


@task
def pipeline_subject(c, subject, slurm=False, n_jobs=-1):
    """
    Run complete subject-level analysis pipeline.

    This runs subject-level GLM for all conditions and generates visualizations.

    Args:
        subject: Subject ID (e.g., sub-01)
        slurm: If True, submit each stage to SLURM
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
    """
    print(f"\n{'='*60}")
    print(f"Running subject-level pipeline for {subject}")
    print(f"{'='*60}\n")

    print(f"\n[1/2] Subject-level GLM for {len(CONDITIONS)} conditions...")
    for cond in CONDITIONS:
        glm_subject_level(c, subject, cond, slurm=slurm, n_jobs=n_jobs)

    print("\n[2/2] Subject-level visualizations...")
    viz_subject_level(c, slurm=slurm)

    print(f"\n{'='*60}")
    print(f"Subject pipeline complete for {subject}!")
    print(f"{'='*60}\n")


# =============================================================================
# Environment Setup Tasks
# =============================================================================

@task
def setup_env(c):
    """Install Python dependencies from requirements.txt."""
    print("Installing dependencies from requirements.txt...")
    c.run("pip install -r requirements.txt")


@task
def setup_airoh(c):
    """Install airoh and invoke for task automation."""
    print("Installing airoh and invoke...")
    c.run("pip install airoh invoke")


@task
def info(c):
    """Display configuration and environment information."""
    print("\n" + "="*60)
    print("Shinobi fMRI Analysis Pipeline Configuration")
    print("="*60)
    print(f"\nProject Root:     {PROJECT_ROOT}")
    print(f"Data Path:        {DATA_PATH}")
    print(f"Python (local):   {PYTHON_BIN}")
    print(f"Python (SLURM):   {SLURM_PYTHON_BIN}")
    print(f"\nSubjects:         {', '.join(SUBJECTS)}")
    print(f"Conditions:       {', '.join(CONDITIONS)}")
    print(f"\nAiroh Available:  {AIROH_AVAILABLE}")

    # Count available data
    try:
        sub_ses_pairs = get_subject_sessions(DATA_PATH)
        runs = get_all_runs(DATA_PATH)
        print(f"\nAvailable Sessions: {len(sub_ses_pairs)}")
        print(f"Available Runs:     {len(runs)}")
    except Exception as e:
        print(f"\nCould not scan data directory: {e}")

    print("\n" + "="*60 + "\n")


# =============================================================================
# Build Task Collections
# =============================================================================

# Create namespace and add tasks
namespace = Collection()

# GLM tasks
glm_collection = Collection('glm')
glm_collection.add_task(glm_run_level, name='run-level')
glm_collection.add_task(glm_session_level, name='session-level')
glm_collection.add_task(glm_subject_level, name='subject-level')
namespace.add_collection(glm_collection)

# MVPA tasks
mvpa_collection = Collection('mvpa')
mvpa_collection.add_task(mvpa_session_level, name='session-level')
namespace.add_collection(mvpa_collection)

# Correlation tasks
corr_collection = Collection('corr')
corr_collection.add_task(correlations, name='compute')
namespace.add_collection(corr_collection)

# Visualization tasks
viz_collection = Collection('viz')
viz_collection.add_task(viz_run_level, name='run-level')
viz_collection.add_task(viz_session_level, name='session-level')
viz_collection.add_task(viz_subject_level, name='subject-level')
viz_collection.add_task(viz_annotation_panels, name='annotation-panels')
viz_collection.add_task(viz_beta_correlations, name='beta-correlations')
viz_collection.add_task(viz_regressor_correlations, name='regressor-correlations')
namespace.add_collection(viz_collection)

# Pipeline tasks
pipeline_collection = Collection('pipeline')
pipeline_collection.add_task(pipeline_full, name='full')
pipeline_collection.add_task(pipeline_subject, name='subject')
namespace.add_collection(pipeline_collection)

# Setup and utility tasks
setup_collection = Collection('setup')
setup_collection.add_task(setup_env, name='env')
setup_collection.add_task(setup_airoh, name='airoh')
namespace.add_collection(setup_collection)

# Diagnostic tasks
diagnostic_collection = Collection('diag')
diagnostic_collection.add_task(diagnose_missing_zmaps, name='missing-zmaps')
namespace.add_collection(diagnostic_collection)

# Top-level utility tasks
namespace.add_task(info)

# Add airoh utilities if available
if AIROH_AVAILABLE:
    namespace.add_collection(utils, name="airoh-utils")
