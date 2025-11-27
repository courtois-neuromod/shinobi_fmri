"""
Shinobi fMRI Analysis Pipeline - Task Automation with Airoh/Invoke

This file provides a unified interface for running the shinobi_fmri analysis pipeline,
supporting both local execution and SLURM-based cluster computing.

Usage:
    invoke --list                           # Show all available tasks
    invoke glm.run-level --subject sub-01 --session ses-001    # Run locally
    invoke glm.run-level --subject sub-01 --session ses-001 --slurm  # Submit to SLURM
    invoke batch.glm-run-level              # Run all subjects/sessions locally
    invoke batch.glm-run-level --slurm      # Submit all to SLURM
"""

from invoke import task, Collection
import os
import os.path as op
import glob

# Try to import airoh utilities (optional)
try:
    from airoh import utils, containers
    AIROH_AVAILABLE = True
except ImportError:
    AIROH_AVAILABLE = False
    print("Note: airoh not installed. Install with 'pip install airoh invoke' for additional utilities.")

# Try to import shinobi_behav for DATA_PATH
try:
    from shinobi_behav import DATA_PATH
except ImportError:
    DATA_PATH = "/home/hyruuk/scratch/data"
    print(f"Warning: shinobi_behav not found. Using default DATA_PATH: {DATA_PATH}")


# =============================================================================
# Configuration
# =============================================================================

# Python environment for local execution
PYTHON_BIN = "python"  # Uses current environment

# Python environment for SLURM execution
SLURM_PYTHON_BIN = "/home/hyruuk/python_envs/shinobi/bin/python"

# Project paths
PROJECT_ROOT = op.dirname(op.abspath(__file__))
SHINOBI_FMRI_DIR = op.join(PROJECT_ROOT, "shinobi_fmri")
SLURM_DIR = op.join(PROJECT_ROOT, "slurm")

# Analysis subjects and conditions
SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
CONDITIONS = ['HIT', 'JUMP', 'DOWN', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP']


# =============================================================================
# Helper Functions
# =============================================================================

def get_subject_sessions(data_path, subject=None):
    """Get all available subject-session combinations from fMRIPrep output."""
    fmriprep_dir = op.join(data_path, "shinobi.fmriprep")

    if subject:
        subjects = [subject]
    else:
        subjects = [d for d in os.listdir(fmriprep_dir) if d.startswith('sub-')]

    sub_ses_pairs = []
    for sub in subjects:
        sub_dir = op.join(fmriprep_dir, sub)
        if op.exists(sub_dir):
            sessions = [d for d in os.listdir(sub_dir) if d.startswith('ses-')]
            for ses in sessions:
                sub_ses_pairs.append((sub, ses))

    return sub_ses_pairs


def get_all_runs(data_path):
    """Get all available runs (subject-session pairs) from fMRIPrep output."""
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

    return runs


# =============================================================================
# GLM Analysis Tasks
# =============================================================================

@task
def glm_run_level(c, subject, session, slurm=False):
    """
    Run run-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01)
        session: Session ID (e.g., ses-001)
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_run_level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_run-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {session}"
        print(f"Submitting to SLURM: {subject} {session}")
    else:
        cmd = f"{PYTHON_BIN} {script} --subject {subject} --session {session}"
        print(f"Running locally: {subject} {session}")

    c.run(cmd)


@task
def glm_session_level(c, subject, session, slurm=False):
    """
    Run session-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01)
        session: Session ID (e.g., ses-001)
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_session_level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_session-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {session}"
        print(f"Submitting to SLURM: {subject} {session}")
    else:
        cmd = f"{PYTHON_BIN} {script} --subject {subject} --session {session}"
        print(f"Running locally: {subject} {session}")

    c.run(cmd)


@task
def glm_subject_level(c, subject, condition, slurm=False):
    """
    Run subject-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01)
        condition: Condition/contrast name (e.g., HIT, JUMP, Kill)
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_subject_level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_subject-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {condition}"
        print(f"Submitting to SLURM: {subject} {condition}")
    else:
        cmd = f"{PYTHON_BIN} {script} --subject {subject} --condition {condition}"
        print(f"Running locally: {subject} {condition}")

    c.run(cmd)


# =============================================================================
# MVPA Tasks
# =============================================================================

@task
def mvpa_session_level(c, subject, task, perm_index=0, slurm=False):
    """
    Run session-level MVPA analysis.

    Args:
        subject: Subject ID (e.g., sub-01)
        task: Task name
        perm_index: Permutation index for significance testing
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "mvpa", "compute_ses-level_with_hcp.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_mvpa_ses-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {task} {perm_index}"
        print(f"Submitting to SLURM: {subject} {task} perm={perm_index}")
    else:
        cmd = f"{PYTHON_BIN} {script} -s {subject} --task {task} --perm-index {perm_index}"
        print(f"Running locally: {subject} {task} perm={perm_index}")

    c.run(cmd)


# =============================================================================
# Correlation Analysis Tasks
# =============================================================================

@task
def correlations(c, chunk_start=0, chunk_size=100, n_jobs=40, slurm=False):
    """
    Compute correlation matrices with HCP data.

    Args:
        chunk_start: Starting index for chunked processing
        chunk_size: Number of maps per chunk
        n_jobs: Number of parallel jobs
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "correlations", "session_corrmat_with_hcp.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_corrmat_chunk.sh")
        cmd = f"sbatch {slurm_script} {chunk_start}"
        print(f"Submitting to SLURM: chunk_start={chunk_start}")
    else:
        cmd = f"{PYTHON_BIN} {script} --chunk-start {chunk_start} --chunk-size {chunk_size} --n-jobs {n_jobs}"
        print(f"Running locally: chunk=[{chunk_start}:{chunk_start+chunk_size}], n_jobs={n_jobs}")

    c.run(cmd)


# =============================================================================
# Visualization Tasks
# =============================================================================

@task
def viz_run_level(c, subject, condition, slurm=False):
    """
    Generate run-level visualizations.

    Args:
        subject: Subject ID (e.g., sub-01)
        condition: Condition/contrast name
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_run-level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-run-level.sh")
        cmd = f"sbatch {slurm_script} {subject} {condition}"
        print(f"Submitting to SLURM: {subject} {condition}")
    else:
        cmd = f"{PYTHON_BIN} {script} -s {subject} -c {condition}"
        print(f"Running locally: {subject} {condition}")

    c.run(cmd)


@task
def viz_session_level(c, slurm=False):
    """
    Generate session-level visualizations.

    Args:
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_session-level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-sesslevel.sh")
        cmd = f"sbatch {slurm_script}"
        print("Submitting to SLURM")
    else:
        cmd = f"{PYTHON_BIN} {script}"
        print("Running locally")

    c.run(cmd)


@task
def viz_subject_level(c, slurm=False):
    """
    Generate subject-level visualizations.

    Args:
        slurm: If True, submit to SLURM cluster
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_subject-level.py")

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_viz-sub-level.sh")
        cmd = f"sbatch {slurm_script}"
        print("Submitting to SLURM")
    else:
        cmd = f"{PYTHON_BIN} {script}"
        print("Running locally")

    c.run(cmd)


@task
def viz_annotation_panels(c, condition=None, conditions=None, skip_individual=False, skip_panels=False, skip_pdf=False):
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

    cmd = ' '.join(cmd_parts)
    print(f"Generating annotation panels...")
    if condition:
        print(f"  Condition: {condition}")
    elif conditions:
        print(f"  Conditions: {conditions}")
    else:
        print(f"  Processing all default conditions")

    c.run(cmd)


# =============================================================================
# Batch Processing Tasks
# =============================================================================

@task
def batch_glm_run_level(c, slurm=False, subject=None):
    """
    Run run-level GLM for all available subjects/sessions.

    Args:
        slurm: If True, submit all jobs to SLURM
        subject: Optional: process only this subject
    """
    runs = get_all_runs(DATA_PATH)

    if subject:
        runs = [(s, ses) for s, ses in runs if s == subject]

    print(f"Processing {len(runs)} runs...")
    for sub, ses in runs:
        glm_run_level(c, sub, ses, slurm=slurm)


@task
def batch_glm_session_level(c, slurm=False, subject=None):
    """
    Run session-level GLM for all available subjects/sessions.

    Args:
        slurm: If True, submit all jobs to SLURM
        subject: Optional: process only this subject
    """
    sub_ses_pairs = get_subject_sessions(DATA_PATH, subject=subject)

    print(f"Processing {len(sub_ses_pairs)} sessions...")
    for sub, ses in sub_ses_pairs:
        glm_session_level(c, sub, ses, slurm=slurm)


@task
def batch_glm_subject_level(c, slurm=False, subjects=None, conditions=None):
    """
    Run subject-level GLM for all subjects and conditions.

    Args:
        slurm: If True, submit all jobs to SLURM
        subjects: Optional: comma-separated list of subjects (e.g., "sub-01,sub-02")
        conditions: Optional: comma-separated list of conditions (e.g., "HIT,JUMP")
    """
    subs = subjects.split(',') if subjects else SUBJECTS
    conds = conditions.split(',') if conditions else CONDITIONS

    print(f"Processing {len(subs)} subjects x {len(conds)} conditions = {len(subs)*len(conds)} jobs...")
    for sub in subs:
        for cond in conds:
            glm_subject_level(c, sub, cond, slurm=slurm)


@task
def batch_correlations(c, num_jobs=100, chunk_size=100, slurm=False):
    """
    Submit multiple correlation computation jobs with chunking.

    Args:
        num_jobs: Number of jobs to submit
        chunk_size: Number of maps per chunk
        slurm: If True, submit to SLURM
    """
    print(f"Submitting {num_jobs} correlation jobs...")
    for job_idx in range(num_jobs):
        chunk_start = job_idx * chunk_size
        correlations(c, chunk_start=chunk_start, chunk_size=chunk_size, slurm=slurm)


# =============================================================================
# Full Pipeline Tasks
# =============================================================================

@task
def pipeline_full(c, subject, session, slurm=False):
    """
    Run complete analysis pipeline for a single subject/session.

    This runs: run-level GLM -> session-level GLM -> visualizations

    Args:
        subject: Subject ID (e.g., sub-01)
        session: Session ID (e.g., ses-001)
        slurm: If True, submit each stage to SLURM
    """
    print(f"\n{'='*60}")
    print(f"Running full pipeline for {subject} {session}")
    print(f"{'='*60}\n")

    print("\n[1/3] Run-level GLM...")
    glm_run_level(c, subject, session, slurm=slurm)

    print("\n[2/3] Session-level GLM...")
    glm_session_level(c, subject, session, slurm=slurm)

    print("\n[3/3] Visualizations...")
    viz_session_level(c, slurm=slurm)

    print(f"\n{'='*60}")
    print(f"Pipeline complete for {subject} {session}!")
    print(f"{'='*60}\n")


@task
def pipeline_subject(c, subject, slurm=False):
    """
    Run complete subject-level analysis pipeline.

    This runs subject-level GLM for all conditions and generates visualizations.

    Args:
        subject: Subject ID (e.g., sub-01)
        slurm: If True, submit each stage to SLURM
    """
    print(f"\n{'='*60}")
    print(f"Running subject-level pipeline for {subject}")
    print(f"{'='*60}\n")

    print(f"\n[1/2] Subject-level GLM for {len(CONDITIONS)} conditions...")
    for cond in CONDITIONS:
        glm_subject_level(c, subject, cond, slurm=slurm)

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
namespace.add_collection(viz_collection)

# Batch processing tasks
batch_collection = Collection('batch')
batch_collection.add_task(batch_glm_run_level, name='glm-run-level')
batch_collection.add_task(batch_glm_session_level, name='glm-session-level')
batch_collection.add_task(batch_glm_subject_level, name='glm-subject-level')
batch_collection.add_task(batch_correlations, name='correlations')
namespace.add_collection(batch_collection)

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

# Top-level utility tasks
namespace.add_task(info)

# Add airoh utilities if available
if AIROH_AVAILABLE:
    namespace.add_collection(utils, name="airoh-utils")
