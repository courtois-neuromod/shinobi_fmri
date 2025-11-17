#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_corrmat
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

/home/hyruuk/python_envs/shinobi/bin/python /home/hyruuk/GitHub/shinobi_fmri/shinobi_fmri/correlations/session_corrmat.py
