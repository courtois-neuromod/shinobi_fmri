#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=shinobi_script
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

/home/hyruuk/python_envs/shinobi_env/bin/python $1
