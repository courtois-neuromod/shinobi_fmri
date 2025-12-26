#!/usr/bin/env python3
"""
Batch launcher for MVPA permutation testing.
Distributes permutations across multiple SLURM jobs.
"""
import os
import subprocess
import argparse

# Default configuration
N_PERMUTATIONS = 1000
PERMS_PER_JOB = 50  # Each job handles 50 permutations
SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']


def submit_permutation_jobs(subjects, n_permutations, perms_per_job, screening=20, n_jobs=40, dry_run=False):
    """
    Submit SLURM jobs for permutation testing.

    Args:
        subjects: List of subjects to process
        n_permutations: Total number of permutations
        perms_per_job: Number of permutations per SLURM job
        screening: Screening percentile
        n_jobs: Number of parallel jobs for each decoder fit
        dry_run: If True, print commands without submitting
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slurm_script = os.path.join(script_dir, "subm_mvpa_permutation.sh")

    job_ids = []

    for subject in subjects:
        print(f"\n=== Submitting permutation jobs for {subject} ===")

        # Calculate number of jobs needed
        n_jobs_needed = (n_permutations + perms_per_job - 1) // perms_per_job

        for job_idx in range(n_jobs_needed):
            perm_start = job_idx * perms_per_job
            perm_end = min((job_idx + 1) * perms_per_job, n_permutations)

            # Construct SLURM command
            cmd = [
                "sbatch",
                slurm_script,
                subject,
                str(n_permutations),
                str(perm_start),
                str(perm_end),
                str(screening),
                str(n_jobs)
            ]

            if dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Extract job ID from output
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    print(f"  Submitted job {job_id}: perms {perm_start}-{perm_end-1}")
                else:
                    print(f"  ERROR submitting job: {result.stderr}")

        print(f"  Total: {n_jobs_needed} jobs for {n_permutations} permutations")

    if not dry_run:
        print(f"\n=== Submitted {len(job_ids)} total jobs ===")
        print(f"Monitor with: squeue -u $USER")

    return job_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch submit MVPA permutation jobs")
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS,
                        help=f"Subjects to process (default: {SUBJECTS})")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Total number of permutations (default: {N_PERMUTATIONS})")
    parser.add_argument("--perms-per-job", type=int, default=PERMS_PER_JOB,
                        help=f"Permutations per SLURM job (default: {PERMS_PER_JOB})")
    parser.add_argument("--screening", type=int, default=20,
                        help="Screening percentile (default: 20)")
    parser.add_argument("--n-jobs", type=int, default=40,
                        help="CPUs per decoder fit (default: 40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")

    args = parser.parse_args()

    submit_permutation_jobs(
        subjects=args.subjects,
        n_permutations=args.n_permutations,
        perms_per_job=args.perms_per_job,
        screening=args.screening,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run
    )
