#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_subjlevel
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

/home/hyruuk/python_envs/shinobi_env/bin/python /project/rrg-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/visualization/viz_session-level.py -s $1 -c $2
