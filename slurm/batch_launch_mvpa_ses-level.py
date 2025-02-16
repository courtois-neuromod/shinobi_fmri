import os
import os.path as op
import glob
from shinobi_behav import DATA_PATH

for sub in ["sub-01", "sub-02", "sub-04", "sub-06"]:
        os.system(f"sbatch ./slurm/subm_mvpa_ses-level.sh {sub} classif")
        for perm_index in range(0, 1000, 10):
                os.system(f"sbatch ./slurm/subm_mvpa_ses-level.sh {sub} perm {perm_index}")
