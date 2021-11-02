#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_secondlevel_fmricontrast
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

/home/hyruuk/python_envs/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/glm_subjectlevel_contrast.py
