#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/mvpa/compute_subject_level.py -s $1
