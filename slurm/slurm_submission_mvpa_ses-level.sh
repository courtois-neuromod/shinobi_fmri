#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/notebooks/test_mvpa.py
