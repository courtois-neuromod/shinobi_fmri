#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_sesslevel
#SBATCH --output=logfiles/%x/%x_%j.out
#SBATCH --error=logfiles/%x/%x_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Create log directory
mkdir -p logfiles/shi_sesslevel

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/shinobi_fmri/shinobi_fmri/glm/compute_session_level.py --subject $1 --session $2
