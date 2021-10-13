#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_firstlevel_fmricontrast
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

/home/hyruuk/python_envs/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/glm_runlevel_contrast.py -s $1 -c $2
