#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=shi_subjlevel
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/visualization/viz_session-level.py
