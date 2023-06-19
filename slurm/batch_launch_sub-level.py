import os
import os.path as op
import glob
from shinobi_behav import DATA_PATH

for sub in ["sub-01", "sub-02", "sub-04", "sub-06"]:
    for condition in ['HIT', 'JUMP', 'DOWN', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP', 'lvl1', 'lvl4', 'lvl5']:
        os.system(f"sbatch ./slurm/slurm_submission_subject-level.sh {sub} {condition}")
        if 'level' not in condition:
            for level in ['lvl1', 'lvl4', 'lvl5']:
                os.system(f"sbatch ./slurm/slurm_submission_subject-level.sh {sub} {condition}X{level}")