#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=6:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/mvpa/compute_ses-level_with_hcp.py -s $1 --task $2 --perm-index $3
