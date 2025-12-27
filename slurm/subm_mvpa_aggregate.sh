#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=1:00:00
#SBATCH --job-name=shi_mvpa_agg
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Arguments from task launcher:
# $1 = subject (e.g., sub-01)
# $2 = n_permutations (total number)
# $3 = screening (percentile)

SUBJECT=$1
N_PERM=$2
SCREENING=${3:-20}

# Create log directory
mkdir -p logfiles/shi_mvpa_agg

echo "=========================================="
echo "MVPA Permutation Aggregation"
echo "=========================================="
echo "Subject:      $SUBJECT"
echo "Permutations: $N_PERM"
echo "Screening:    $SCREENING%"
echo "=========================================="

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/mvpa/aggregate_permutations.py \
    --subject $SUBJECT \
    --n-permutations $N_PERM \
    --screening $SCREENING

echo "=========================================="
echo "Aggregation completed"
echo "=========================================="
