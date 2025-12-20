#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_viz_runlvl
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Create log directory
mkdir -p logfiles/shi_viz_runlvl

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/neuromod/shinobi_fmri/shinobi_fmri/visualization/viz_run-level.py --subject $1 --condition $2
