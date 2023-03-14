#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shi_mvpa_seslvl
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

<<<<<<< HEAD
/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/notebooks/test_mvpa.py
=======
/home/hyruuk/python_envs/shinobi_env/bin/python /home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri/shinobi_fmri/notebooks/test_mvpa.py
>>>>>>> a2a93cf4ac0241709827fff43da7285d6a7e3542
