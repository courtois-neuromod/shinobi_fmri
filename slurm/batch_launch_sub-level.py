import os
import os.path as op
import glob
from shinobi_behav import DATA_PATH

for sub in ["sub-01", "sub-02", "sub-04", "sub-06"]:
    for condition in ['HIT', 'JUMP', 'DOWN', 'HealthGain', 'HealthLoss', 'Kill', 'LEFT', 'RIGHT', 'UP', 'HIT+JUMP-RIGHT-LEFT-UP-DOWN', 'RIGHT+LEFT+UP+DOWN-HIT-JUMP']:
        os.system(f"sbatch ./slurm/slurm_submission_subject-level.sh {sub} {condition}")