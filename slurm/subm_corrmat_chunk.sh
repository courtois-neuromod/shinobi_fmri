#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=03:00:00
#SBATCH --job-name=shi_corr_chunk
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

CHUNK_START=${1:-0}
CHUNK_SIZE=100

/home/hyruuk/python_envs/shinobi/bin/python \
    /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/correlations/compute_beta_correlations.py \
    --chunk-size "${CHUNK_SIZE}" \
    --chunk-start "${CHUNK_START}" \
    --n-jobs 40
