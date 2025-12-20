#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_runlevel
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Create log directory
mkdir -p logfiles/shi_runlevel

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/shinobi_fmri/shinobi_fmri/glm/compute_run_level.py -s $1 -ses $2
