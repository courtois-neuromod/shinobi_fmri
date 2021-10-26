#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_secondlevel_fmricontrast
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

/lustre03/project/6003287/hyruuk/.virtualenvs/hyruuk_shinobi_behav/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/glm_subjectlevel_contrast.py
