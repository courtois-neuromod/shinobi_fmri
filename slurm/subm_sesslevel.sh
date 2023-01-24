#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=4:00:00
#SBATCH --job-name=shi_sesslevel
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /project/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/compute_seslevel.py --subject $1 --session $2
