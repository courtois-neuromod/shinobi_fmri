#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_runlevel
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/glm/compute_run_level.py -s $1 -ses $2 -r $3
