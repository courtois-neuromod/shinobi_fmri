import os
import os.path as op
import glob
from shinobi_behav import DATA_PATH

for sub in ["sub-01", "sub-02", "sub-04", "sub-06"]:
        os.system(f"sbatch ./slurm/subm_viz-run-level.sh {sub}")
