for COND in Jump Hit HealthLoss; do
for SUB in 01 02 04 06; do
sbatch ./slurm/subm_runlevel.sh $SUB $COND
done
done
