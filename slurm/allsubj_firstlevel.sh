for SUB in 01 02 04 06; do
sbatch ./slurm/subm_firstlevel.sh $SUB
done
