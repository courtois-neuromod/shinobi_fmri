#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_mvpa_perm
#SBATCH --output=logfiles/%x/%x_%A_%a.out
#SBATCH --error=logfiles/%x/%x_%A_%a.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Arguments from batch launcher:
# $1 = subject (e.g., sub-01)
# $2 = n_permutations (total number)
# $3 = perm_start (starting index for this job)
# $4 = perm_end (ending index for this job)
# $5 = screening (percentile)
# $6 = n_jobs (CPUs)

SUBJECT=$1
N_PERM=$2
PERM_START=$3
PERM_END=$4
SCREENING=${5:-20}
N_JOBS=${6:-40}

# Create log directory
mkdir -p logfiles/shi_mvpa_perm

echo "=========================================="
echo "MVPA Permutation Testing"
echo "=========================================="
echo "Subject:      $SUBJECT"
echo "Permutations: $PERM_START to $((PERM_END-1)) (out of $N_PERM total)"
echo "Screening:    $SCREENING%"
echo "CPUs:         $N_JOBS"
echo "=========================================="

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/mvpa/compute_mvpa.py \
    --subject $SUBJECT \
    --n-permutations $N_PERM \
    --perm-start $PERM_START \
    --perm-end $PERM_END \
    --screening $SCREENING \
    --n-jobs $N_JOBS \
    -v

echo "=========================================="
echo "Permutation job completed"
echo "=========================================="
