#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=01:00:00
#SBATCH --job-name=shi_subjlevel
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/shinobi_fmri/shinobi_fmri/glm/compute_subject_level.py --subject $1 --condition $2
