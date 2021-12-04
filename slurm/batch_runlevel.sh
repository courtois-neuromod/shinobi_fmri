for SUB in 01 02 04 06; do
sbatch ./slurm/subm_runlevel.sh $SUB
done
