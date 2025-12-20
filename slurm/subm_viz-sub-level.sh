#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_viz_sublvl
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

# Create log directory
mkdir -p logfiles/shi_viz_sublvl

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/visualization/viz_subject-level.py
