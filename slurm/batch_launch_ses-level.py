import os
import os.path as op
import glob
from shinobi_fmri.config import DATA_PATH

filelist = os.listdir(op.join(DATA_PATH, "shinobi.fmriprep"))
for name in filelist:
    if len(name) == 6:
        if "sub-" in name:
            sub = name
            seslist = os.listdir(op.join(DATA_PATH, "shinobi.fmriprep", name))
            seslist = [x for x in seslist if "ses-" in x]
            for ses in seslist:
                os.system(f"sbatch ./slurm/subm_session-level.sh {sub} {ses}")
