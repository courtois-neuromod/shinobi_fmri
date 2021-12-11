for FROM in run session; do
for COND in Jump Hit Kill; do
for SUB in 01 02 04 06; do
sbatch ./slurm/subm_subjlevel.sh $SUB $COND $FROM
done
done
done
