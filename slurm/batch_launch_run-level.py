import os
import os.path as op
import glob
from shinobi_behav import DATA_PATH

filelist = glob.glob(op.join(DATA_PATH, "shinobi.fmriprep", "*", "*", "*", "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))

for file in filelist:
    sub = file.split("/")[-1].split("_")[0]
    ses = file.split("/")[-1].split("_")[1]
    os.system(f"sbatch ./slurm/slurm_submission_run-level.sh {sub} {ses}")