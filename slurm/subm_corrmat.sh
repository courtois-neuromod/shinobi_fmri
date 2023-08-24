#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=48:00:00
#SBATCH --job-name=shinobi_corrmat
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/correlations/session_corrmat.py
