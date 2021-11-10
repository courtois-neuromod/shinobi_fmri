for COND in Jump Hit HealthLoss Kill; do
for SUB in 01 02 04 06; do
sbatch ./slurm/subm_subjlevel.sh $SUB $COND
done
done
