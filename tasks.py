"""
Shinobi fMRI Analysis Pipeline - Task Automation with Airoh/Invoke

This file provides a unified interface for running the shinobi_fmri analysis pipeline,
supporting both local execution and SLURM-based cluster computing.

Usage:
    invoke --list                                               # Show all available tasks
    invoke glm.session-level --subject sub-01 --session ses-001 # Run single subject/session locally
    invoke glm.session-level --slurm                            # Submit all to SLURM (batch mode)
"""

from invoke import task, Collection
import os
import os.path as op
import glob

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import configuration
try:
    from shinobi_fmri.config import DATA_PATH, FIG_PATH, SUBJECTS, CONDITIONS, PYTHON_BIN, SLURM_PYTHON_BIN
except ImportError:
    # Fallback to environment variables and defaults
    DATA_PATH = os.getenv("SHINOBI_DATA_PATH", "/home/hyruuk/scratch/data")
    FIG_PATH = os.getenv("SHINOBI_FIG_PATH", "reports/figures")
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
        low_level_flag = "--low-level-confs" if low_level_confs else ""
        cmd = f"sbatch {slurm_script} {subject} {session} {low_level_flag}"
        print(f"Submitting to SLURM: {subject} {session}{' (low-level-confs)' if low_level_confs else ''}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def glm_subject_level(c, subject=None, condition=None, slurm=False, n_jobs=-1, verbose=0, log_dir=None, low_level_confs=False):
    """
    Run subject-level GLM analysis.

    Args:
        subject: Subject ID (e.g., sub-01) - if None, process all subjects
        condition: Condition/contrast name (e.g., HIT, JUMP, Kill) - if None, process all conditions
        slurm: If True, submit to SLURM cluster (single job or batch mode depending on subject/condition)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        verbose: Verbosity level
        log_dir: Custom log directory
        low_level_confs: If True, use session-level maps from GLM with low-level confounds (processed_low-level/ directory)
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
                glm_subject_level(c, subject=sub, condition=cond, slurm=slurm, n_jobs=n_jobs, verbose=verbose, log_dir=log_dir, low_level_confs=low_level_confs)
        return

    # Process single subject/condition
    script = op.join(SHINOBI_FMRI_DIR, "glm", "compute_subject_level.py")

    args = f"--subject {subjects[0]} --condition {conditions[0]}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"
    if low_level_confs:
        args += " --low-level-confs"

    if slurm:
        slurm_script = op.join(SLURM_DIR, "subm_subject-level.sh")
        low_level_flag = "--low-level-confs" if low_level_confs else ""
        cmd = f"sbatch {slurm_script} {subjects[0]} {conditions[0]} {low_level_flag}"
        print(f"Submitting to SLURM: {subjects[0]} {conditions[0]}{' (low-level-confs)' if low_level_confs else ''}")
    else:
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: {cmd}")

    c.run(cmd)


@task
def glm_apply_cluster_correction(c, level="both", subject=None, session=None, threshold=None, alpha=0.05, overwrite=False, verbose=0, log_dir=None):
    """
    Apply cluster-level FWE correction to existing z-maps.

    Takes raw uncorrected z-maps and applies cluster-level family-wise error
    correction, saving properly thresholded z-maps (not p-value maps).

    Useful for:
    - Re-running correction with different thresholds
    - Applying correction to z-maps generated before this fix
    - Fixing p-value maps to proper thresholded z-maps

    Args:
        level: Analysis level to process ('subject', 'session', or 'both')
        subject: Process specific subject only (default: all subjects)
        session: Process specific session only (session-level only, default: all sessions)
        threshold: Cluster-forming threshold (default: from config - 2.3 for subject/session)
        alpha: Family-wise error rate (default: 0.05)
        overwrite: Overwrite existing corrected z-maps
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "glm", "apply_cluster_correction.py")

    cmd_parts = [PYTHON_BIN, script, "--level", level]

    if subject:
        cmd_parts.extend(["--subject", subject])
    if session:
        cmd_parts.extend(["--session", session])
    if threshold is not None:
        cmd_parts.extend(["--threshold", str(threshold)])
    if alpha != 0.05:
        cmd_parts.extend(["--alpha", str(alpha)])
    if overwrite:
        cmd_parts.append("--overwrite")
    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(["--log-dir", log_dir])

    cmd = ' '.join(cmd_parts)
    print(f"Applying cluster-level FWE correction ({level}-level)...")
    c.run(cmd)


# =============================================================================
# MVPA Tasks
# =============================================================================

@task
def mvpa_session_level(c, subject=None, n_permutations=1000, perms_per_job=50,
                       screening=20, n_jobs=-1, slurm=False,
                       skip_decoder=False, skip_permutations=False, skip_aggregate=False,
                       verbose=0, log_dir=None):
    """
    Run complete session-level MVPA pipeline: decoder + permutations + aggregation.

    Supports both local (sequential) and SLURM (parallel with job dependencies) execution.

    Args:
        subject: Subject ID (e.g., sub-01). If None, processes all subjects.
        n_permutations: Total number of permutations (default: 1000, set to 0 to skip)
        perms_per_job: Number of permutations per SLURM job (default: 50, ignored for local)
        screening: Feature screening percentile (default: 20)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        slurm: If True, submit to SLURM cluster with job dependencies
        skip_decoder: Skip decoder step (run only permutations/aggregation)
        skip_permutations: Skip permutation testing
        skip_aggregate: Skip aggregation step
        verbose: Verbosity level
        log_dir: Custom log directory

    Examples:
        # Run full pipeline locally for one subject
        invoke mvpa.session-level --subject sub-01

        # Run full pipeline on SLURM for all subjects
        invoke mvpa.session-level --slurm

        # Run only decoder (no permutations)
        invoke mvpa.session-level --subject sub-01 --n-permutations 0

        # Run only permutations and aggregation (decoder already done)
        invoke mvpa.session-level --subject sub-01 --skip-decoder --slurm
    """
    # Determine subjects to process
    if subject is None:
        subjects = SUBJECTS
    else:
        subjects = [subject]

    for sub in subjects:
        print(f"\n{'='*60}")
        print(f"MVPA Pipeline for {sub}")
        print(f"{'='*60}")

        if slurm:
            # ================================================================
            # SLURM mode with job dependencies
            # ================================================================
            job_ids = []

            # Step 1: Submit decoder job
            if not skip_decoder:
                decoder_script = op.join(SLURM_DIR, "subm_mvpa_ses-level.sh")
                # Use --parsable to get just the job ID
                cmd = f"sbatch --parsable {decoder_script} {sub} {screening} {n_jobs}"
                print(f"[1/3] Submitting decoder job for {sub}...")
                result = c.run(cmd, hide=True)
                decoder_job_id = result.stdout.strip()
                job_ids.append(decoder_job_id)
                print(f"  ✓ Decoder job: {decoder_job_id}")

            # Step 2: Submit permutation jobs (if requested)
            perm_job_ids = []
            if n_permutations > 0 and not skip_permutations:
                print(f"[2/3] Submitting permutation jobs for {sub}...")
                n_jobs_needed = (n_permutations + perms_per_job - 1) // perms_per_job

                perm_script = op.join(SLURM_DIR, "subm_mvpa_permutation.sh")
                for job_idx in range(n_jobs_needed):
                    perm_start = job_idx * perms_per_job
                    perm_end = min((job_idx + 1) * perms_per_job, n_permutations)

                    cmd = f"sbatch --parsable {perm_script} {sub} {n_permutations} {perm_start} {perm_end} {screening} {n_jobs}"
                    result = c.run(cmd, hide=True)
                    perm_job_id = result.stdout.strip()
                    perm_job_ids.append(perm_job_id)
                    job_ids.append(perm_job_id)

                print(f"  ✓ Submitted {len(perm_job_ids)} permutation jobs: {perm_job_ids[0]}-{perm_job_ids[-1]}")

            # Step 3: Submit aggregation job with dependencies on ALL previous jobs
            if n_permutations > 0 and not skip_aggregate:
                if not job_ids:
                    print(f"  ⚠ Warning: No jobs to depend on for aggregation. Run decoder and/or permutations first.")
                else:
                    print(f"[3/3] Submitting aggregation job for {sub}...")
                    agg_script = op.join(SLURM_DIR, "subm_mvpa_aggregate.sh")
                    dependency_str = ":".join(job_ids)
                    cmd = f"sbatch --parsable --dependency=afterok:{dependency_str} {agg_script} {sub} {n_permutations} {screening}"
                    result = c.run(cmd, hide=True)
                    agg_job_id = result.stdout.strip()
                    print(f"  ✓ Aggregation job: {agg_job_id} (waits for {len(job_ids)} jobs)")

            print(f"\n{'='*60}")
            print(f"SLURM jobs submitted for {sub}!")
            print(f"Monitor with: squeue -u $USER")
            print(f"{'='*60}\n")

        else:
            # ================================================================
            # Local mode - run sequentially
            # ================================================================

            # Step 1: Run decoder
            if not skip_decoder:
                print(f"[1/3] Running decoder for {sub}...")
                script = op.join(SHINOBI_FMRI_DIR, "mvpa", "compute_mvpa.py")
                args = f"--subject {sub} --screening {screening} --n-jobs {n_jobs}"
                if isinstance(verbose, int) and verbose > 0:
                    args += f" -{'v' * verbose}"
                if log_dir:
                    args += f" --log-dir {log_dir}"
                cmd = f"{PYTHON_BIN} {script} {args}"
                c.run(cmd)
                print(f"  ✓ Decoder complete")

            # Step 2: Run permutations
            if n_permutations > 0 and not skip_permutations:
                print(f"[2/3] Running {n_permutations} permutations for {sub}...")
                script = op.join(SHINOBI_FMRI_DIR, "mvpa", "compute_mvpa.py")
                args = f"--subject {sub} --screening {screening} --n-jobs {n_jobs}"
                args += f" --n-permutations {n_permutations} --perm-start 0 --perm-end {n_permutations}"
                if isinstance(verbose, int) and verbose > 0:
                    args += f" -{'v' * verbose}"
                if log_dir:
                    args += f" --log-dir {log_dir}"
                cmd = f"{PYTHON_BIN} {script} {args}"
                c.run(cmd)
                print(f"  ✓ Permutations complete")

            # Step 3: Aggregate results
            if n_permutations > 0 and not skip_aggregate:
                print(f"[3/3] Aggregating permutation results for {sub}...")
                script = op.join(SHINOBI_FMRI_DIR, "mvpa", "aggregate_permutations.py")
                cmd = f"{PYTHON_BIN} {script} --subject {sub} --n-permutations {n_permutations} --screening {screening}"
                c.run(cmd)
                print(f"  ✓ Aggregation complete")

            print(f"\n{'='*60}")
            print(f"MVPA pipeline complete for {sub}!")
            print(f"{'='*60}\n")




# =============================================================================
# Correlation Analysis Tasks
# =============================================================================

@task
def beta_correlations(c, chunk_start=0, chunk_size=100, n_jobs=-1, slurm=False, verbose=0, log_dir=None, low_level_confs=False):
    """
    Compute beta map correlations with HCP data.

    Args:
        chunk_start: Starting index for chunked processing (only used if not in slurm batch mode)
        chunk_size: Number of maps per chunk (default: 100)
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
        slurm: If True, automatically submit all chunks as SLURM jobs
        verbose: Verbosity level
        log_dir: Custom log directory
        low_level_confs: Use beta maps from GLM with low-level confounds (processed_low-level/ directory)
    """
    script = op.join(SHINOBI_FMRI_DIR, "correlations", "compute_beta_correlations.py")

    args = f"--chunk-size {chunk_size} --n-jobs {n_jobs}"
    if isinstance(verbose, int) and verbose > 0:
        args += f" -{'v' * verbose}"
    if log_dir:
        args += f" --log-dir {log_dir}"
    if low_level_confs:
        args += " --low-level-confs"

    if slurm:
        # Use the script's built-in --slurm mode to batch submit all chunks
        args += " --slurm"
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Submitting all correlation chunks to SLURM...")
    else:
        # Run single chunk locally
        args = f"--chunk-start {chunk_start} {args}"
        cmd = f"{PYTHON_BIN} {script} {args}"
        print(f"Running locally: chunk_start={chunk_start}, chunk_size={chunk_size}")

    c.run(cmd)


@task
def fingerprinting(c, verbose=0, log_dir=None):
    """
    Run fingerprinting analysis on beta maps.

    Assesses whether brain maps are participant-specific by checking if each map's
    most similar map (nearest neighbor) comes from the same subject.

    Args:
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "correlations", "fingerprinting_analysis.py")

    cmd_parts = [PYTHON_BIN, script]

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(["--log-dir", log_dir])

    cmd = ' '.join(cmd_parts)

    print("Running fingerprinting analysis...")
    c.run(cmd)


# =============================================================================
# Visualization Tasks
# =============================================================================

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
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_session_level.py")
    
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
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_subject_level.py")
    
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
def viz_annotation_panels(c, condition=None, conditions=None, skip_individual=False, skip_panels=False, skip_pdf=False, use_corrected_maps=False, force=False, low_level_confs=False, verbose=0, log_dir=None):
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
        use_corrected_maps: Use cluster-corrected z-maps instead of raw maps (default: False)
        force: Force regeneration of images even if they already exist
        low_level_confs: Use results from GLM with low-level confounds (default: False)
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
    if use_corrected_maps:
        cmd_parts.append('--use-corrected-maps')
    if force:
        cmd_parts.append('--force')
    if low_level_confs:
        cmd_parts.append('--low-level-confs')
        
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
def viz_beta_correlations(c, input_path=None, output_path=None, low_level=False, verbose=0, log_dir=None):
    """
    Generate beta correlations figure.

    Args:
        input_path: Path to input .pkl file (default: {DATA_PATH}/processed/beta_correlations.pkl)
        output_path: Path to save the output figure (default: {FIG_PATH}/beta_correlations_plot.png)
        low_level: Use low-level correlation matrix (processed_low-level/ and _low-level.png)
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_beta_correlations.py")

    cmd_parts = [PYTHON_BIN, script]

    if input_path:
        cmd_parts.extend(['--input', input_path])
    if output_path:
        cmd_parts.extend(['--output', output_path])
    
    if low_level:
        cmd_parts.append('--low-level')

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)

    print(f"Generating beta correlations figure...")
    if input_path:
        print(f"  Input: {input_path}")
    else:
        input_loc = "processed_low-level" if low_level else "processed"
        print(f"  Input: Using default from config ({input_loc})")
    
    if output_path:
        print(f"  Output: {output_path}")
    else:
        suffix = "_low-level" if low_level else ""
        print(f"  Output: Using default from config (beta_correlations_plot{suffix}.png)")

    c.run(cmd)


@task
def viz_regressor_correlations(c, subject=None, skip_generation=False, low_level_confs=False, verbose=0, log_dir=None):
    """
    Generate correlation matrices for design matrix regressors.

    Creates design matrices and 2x2 correlation grid showing relationships between
    Shinobi task conditions and psychophysics confounds.

    Args:
        subject: Specific subject to process (default: all subjects)
        skip_generation: Skip design matrix generation, only plot from existing pickle
        low_level_confs: Include low-level confounds (psychophysics and button presses) (default: False)
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_regressor_correlations.py")

    cmd_parts = [PYTHON_BIN, script]

    if subject:
        cmd_parts.extend(['--subject', subject])
    if skip_generation:
        cmd_parts.append('--skip-generation')
    if not low_level_confs:
        cmd_parts.append('--no-low-level-confs')
    else:
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
        print(f"  Including low-level confounds (psychophysics and button presses)")

    c.run(cmd)


@task
def viz_condition_comparison(c, cond1=None, cond2=None, run_all=False, threshold=3.0, use_corrected_maps=False, verbose=0, log_dir=None, output_dir=None):
    """
    Generate condition comparison surface plots.

    Creates surface plots comparing two conditions with three-color overlay:
    - Blue: Significant only for condition 1
    - Red: Significant only for condition 2
    - Purple: Significant for both conditions

    By default (--run-all), generates all predefined comparisons:
    - Kill vs reward (shinobi vs HCP gambling)
    - HealthLoss vs punishment (shinobi vs HCP gambling)
    - RIGHT vs JUMP (shinobi vs shinobi)
    - LEFT vs HIT (shinobi vs shinobi)

    Args:
        cond1: First condition in format "source:condition" (e.g., "shinobi:Kill" or "hcp:reward")
        cond2: Second condition in format "source:condition" (e.g., "shinobi:HealthLoss" or "hcp:punishment")
        run_all: Generate all predefined comparisons (default if no conditions specified)
        threshold: Significance threshold for z-maps (default: 3.0)
        use_corrected_maps: Use cluster-corrected z-maps instead of raw maps (default: False, uses raw maps)
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_dir: Custom log directory
        output_dir: Custom output directory (default: reports/figures/condition_comparison/)
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_condition_comparison.py")

    cmd_parts = [PYTHON_BIN, script]

    # If no conditions specified, default to run-all
    if not cond1 and not cond2 and not run_all:
        run_all = True

    if run_all:
        cmd_parts.append('--run-all')
    else:
        if cond1:
            cmd_parts.extend(['--cond1', cond1])
        if cond2:
            cmd_parts.extend(['--cond2', cond2])

    if threshold != 3.0:
        cmd_parts.extend(['--threshold', str(threshold)])

    if use_corrected_maps:
        cmd_parts.append('--use-corrected-maps')

    if output_dir:
        cmd_parts.extend(['--output-dir', output_dir])

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)

    print(f"Generating condition comparison plots...")
    if run_all:
        print(f"  Generating all predefined comparisons")
    else:
        if cond1:
            print(f"  Condition 1: {cond1}")
        if cond2:
            print(f"  Condition 2: {cond2}")

    c.run(cmd)


@task
def viz_atlas_tables(c, input_dir=None, output_dir=None, cluster_extent=5, voxel_thresh=3.0, direction='both', use_corrected_maps=False, overwrite=False):
    """
    Generate atlas tables for z-maps.

    Args:
        input_dir: Directory containing z-maps (default: {DATA_PATH}/processed/z_maps/subject-level)
        output_dir: Directory to save output tables (default: reports/tables)
        cluster_extent: Minimum cluster size in voxels (default: 5)
        voxel_thresh: Voxel threshold for significance (default: 3.0)
        direction: Direction of the contrast (both, pos, neg)
        use_corrected_maps: Use cluster-corrected z-maps instead of raw maps (default: False, uses raw maps)
        overwrite: Overwrite existing cluster files
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_atlas_tables.py")

    cmd_parts = [PYTHON_BIN, script]

    if input_dir:
        cmd_parts.extend(['--input-dir', input_dir])
    if output_dir:
        cmd_parts.extend(['--output-dir', output_dir])

    cmd_parts.extend(['--cluster-extent', str(cluster_extent)])
    cmd_parts.extend(['--voxel-thresh', str(voxel_thresh)])
    cmd_parts.extend(['--direction', direction])

    if use_corrected_maps:
        cmd_parts.append('--use-corrected-maps')

    if overwrite:
        cmd_parts.append('--overwrite')

    cmd = ' '.join(cmd_parts)
    print(f"Generating atlas tables...")
    c.run(cmd)


@task
def viz_fingerprinting(c, verbose=0, log_dir=None):
    """
    Generate fingerprinting analysis visualizations.

    Creates plots showing subject identification from nearest neighbors,
    within vs between subject correlation distributions, and fingerprinting
    scores by different groupings.

    Args:
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_fingerprinting.py")

    cmd_parts = [PYTHON_BIN, script]

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)

    print("Generating fingerprinting visualizations...")
    c.run(cmd)


@task
def viz_within_subject_correlations(c, verbose=0, log_dir=None):
    """
    Compute and visualize within-subject condition correlations.

    Computes how maps from different conditions correlate with each other
    within each subject, then creates heatmaps and plots demonstrating
    condition specificity.

    Args:
        verbose: Verbosity level
        log_dir: Custom log directory
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_within_subject_correlations.py")

    cmd_parts = [PYTHON_BIN, script]

    if isinstance(verbose, int) and verbose > 0:
        cmd_parts.append(f"-{'v' * verbose}")
    if log_dir:
        cmd_parts.extend(['--log-dir', log_dir])

    cmd = ' '.join(cmd_parts)

    print("Computing and visualizing within-subject condition correlations...")
    c.run(cmd)


@task
def viz_mvpa_confusion_matrices(c, screening=20, output=None):
    """
    Plot MVPA confusion matrices for all subjects.

    Creates a 2x2 grid visualization showing classification confusion matrices
    with task icons and separation styling for clear interpretation.

    Args:
        screening: Screening percentile used (default: 20)
        output: Output path for figure (default: auto-generated in reports/figures_raw/)
    """
    script = op.join(SHINOBI_FMRI_DIR, "visualization", "viz_mvpa_confusion_matrices.py")

    cmd = f"{PYTHON_BIN} {script} --screening {screening} --no-show"
    if output:
        cmd += f" --output {output}"

    print(f"Plotting MVPA confusion matrices: {cmd}")
    c.run(cmd)


# =============================================================================
# Full Pipeline Tasks
# =============================================================================

@task
def pipeline_full(c, subject, session, slurm=False, n_jobs=-1):
    """
    Run complete analysis pipeline for a single subject/session.

    This runs: session-level GLM -> visualizations

    Args:
        subject: Subject ID (e.g., sub-01)
        session: Session ID (e.g., ses-001)
        slurm: If True, submit each stage to SLURM
        n_jobs: Number of parallel jobs (default: -1 = all CPU cores)
    """
    print(f"\n{'='*60}")
    print(f"Running full pipeline for {subject} {session}")
    print(f"{'='*60}\n")

    print("\n[1/2] Session-level GLM...")
    glm_session_level(c, subject, session, slurm=slurm, n_jobs=n_jobs)

    print("\n[2/2] Visualizations...")
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
# Validation Tasks
# =============================================================================

@task
def validate_outputs(c, subject=None, analysis_type='all', check_integrity=False,
                    output=None, verbose=False, log_dir=None):
    """
    Validate shinobi_fmri pipeline outputs for completeness.

    Checks all expected analysis outputs exist and optionally validates file integrity.

    Args:
        subject: Specific subject to validate (e.g., sub-01). Default: all subjects
        analysis_type: Type of analysis to validate
                      (glm_run, glm_session, glm_subject, mvpa, correlations, figures, all)
        check_integrity: If True, validate file contents (slower but thorough)
        output: Path to save detailed JSON report (optional)
        verbose: If True, enable verbose output (INFO level)
        log_dir: Custom log directory

    Examples:
        # Validate everything
        invoke validate.outputs

        # Validate specific subject
        invoke validate.outputs --subject sub-01

        # Validate only GLM outputs
        invoke validate.outputs --analysis-type glm_run

        # Validate figures and visualizations
        invoke validate.outputs --analysis-type figures

        # Full integrity check with detailed report
        invoke validate.outputs --check-integrity --output validation_report.json --verbose
    """
    script = op.join(PROJECT_ROOT, "tests", "validate_outputs.py")

    args = []
    if subject:
        args.append(f"--subject {subject}")
    if analysis_type != 'all':
        args.append(f"--analysis-type {analysis_type}")
    if check_integrity:
        args.append("--check-integrity")
    if output:
        args.append(f"--output {output}")
    if verbose:
        args.append("-v")
    if log_dir:
        args.append(f"--log-dir {log_dir}")

    cmd = f"{PYTHON_BIN} {script} {' '.join(args)}"
    print(f"Running validation: {cmd}")
    c.run(cmd)


# =============================================================================
# Build Task Collections
# =============================================================================

# Create namespace and add tasks
namespace = Collection()

# GLM tasks
glm_collection = Collection('glm')
glm_collection.add_task(glm_session_level, name='session-level')
glm_collection.add_task(glm_subject_level, name='subject-level')
glm_collection.add_task(glm_apply_cluster_correction, name='apply-cluster-correction')
namespace.add_collection(glm_collection)

# MVPA tasks
mvpa_collection = Collection('mvpa')
mvpa_collection.add_task(mvpa_session_level, name='session-level')
namespace.add_collection(mvpa_collection)

# Correlation tasks
corr_collection = Collection('corr')
corr_collection.add_task(beta_correlations, name='beta')
corr_collection.add_task(fingerprinting, name='fingerprinting')
namespace.add_collection(corr_collection)

# Visualization tasks
viz_collection = Collection('viz')
viz_collection.add_task(viz_session_level, name='session-level')
viz_collection.add_task(viz_subject_level, name='subject-level')
viz_collection.add_task(viz_annotation_panels, name='annotation-panels')
viz_collection.add_task(viz_beta_correlations, name='beta-correlations')
viz_collection.add_task(viz_regressor_correlations, name='regressor-correlations')
viz_collection.add_task(viz_condition_comparison, name='condition-comparison')
viz_collection.add_task(viz_atlas_tables, name='atlas-tables')
viz_collection.add_task(viz_fingerprinting, name='fingerprinting')
viz_collection.add_task(viz_within_subject_correlations, name='within-subject-correlations')
viz_collection.add_task(viz_mvpa_confusion_matrices, name='mvpa-confusion-matrices')
namespace.add_collection(viz_collection)

# Pipeline tasks
pipeline_collection = Collection('pipeline')
pipeline_collection.add_task(pipeline_full, name='full')
pipeline_collection.add_task(pipeline_subject, name='subject')
namespace.add_collection(pipeline_collection)

# Validation tasks
validate_collection = Collection('validate')
validate_collection.add_task(validate_outputs, name='outputs')
namespace.add_collection(validate_collection)

# Top-level utility tasks
namespace.add_task(info)
