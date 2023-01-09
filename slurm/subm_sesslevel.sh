#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_sesslevel
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/compute_seslevel.py -s $1 -ses $2
