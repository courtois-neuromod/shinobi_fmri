#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_firstlevel_fmricontrast
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

module load python/3.8.0

/home/hyruuk/python_env/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/NOCONF_glm_sessionlevel_contrast.py -s 01
