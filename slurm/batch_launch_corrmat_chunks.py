import os

NUM_JOBS = 100
CHUNK_SIZE = 100
SUBMISSION_SCRIPT = "./slurm/subm_corrmat_chunk.sh"

for job_idx in range(NUM_JOBS):
    chunk_start = job_idx * CHUNK_SIZE
    os.system(f"sbatch {SUBMISSION_SCRIPT} {chunk_start}")
