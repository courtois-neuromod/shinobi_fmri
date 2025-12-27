#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Arguments from task launcher:
# $1 = subject (e.g., sub-01)
# $2 = screening (percentile, default: 20)
# $3 = n_jobs (CPUs, default: 40)

SUBJECT=$1
SCREENING=${2:-20}
N_JOBS=${3:-40}

# Create log directory
mkdir -p logfiles/shi_mvpa_seslvl

echo "=========================================="
echo "MVPA Session-Level Decoder"
echo "=========================================="
echo "Subject:   $SUBJECT"
echo "Screening: $SCREENING%"
echo "CPUs:      $N_JOBS"
echo "=========================================="

# Run MVPA decoder (no permutations)
/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/mvpa/compute_mvpa.py \
    --subject $SUBJECT \
    --screening $SCREENING \
    --n-jobs $N_JOBS \
    -v

echo "=========================================="
echo "Decoder completed"
echo "=========================================="
