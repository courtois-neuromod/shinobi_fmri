#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_subjlevel
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /project/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/compute_subject_level.py --subject $1 --condition $2