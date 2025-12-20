#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

# Create log directory
mkdir -p logfiles/shi_mvpa_seslvl

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/mvpa/compute_mvpa.py --subject $1 --n_jobs 40
