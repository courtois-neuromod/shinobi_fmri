#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_corrmat
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Create log directory
mkdir -p logfiles/shinobi_corrmat

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/correlations/compute_beta_correlations.py
