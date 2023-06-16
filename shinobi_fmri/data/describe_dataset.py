import pandas as pd
import numpy as np
import os
import shinobi_behav
import glob
import os.path as op
from shinobi_behav import DATA_PATH
import nibabel as nib
import json

def build_dataset_description(dataset_fname):
    eventsfile_list = sorted(glob.glob(op.join(DATA_PATH, "shinobi_released", "shinobi", "*", "*", "*", "*_desc-annotated_events.tsv")))
    sub_list = []
    run_list = []
    ses_list = []
    fmrifile_list = []
    nvol_list = []
    nrep_total_list = []
    nrep_usable_list = []
    nlvl1_list = []
    nlvl4_list = []
    nlvl5_list = []
    nclear_list = []
    nhealthloss_list = []
    nkill_list = []
    for events_file in eventsfile_list:
        usable_reps = 0
        nclear = 0
        nkill = 0
        nhealthloss = 0
        nlvl1 = 0
        nlvl4 = 0
        nlvl5 = 0

        print(events_file)
        # Get general info
        sub = events_file.split('/')[7]
        ses = events_file.split('/')[8]
        run = events_file.split('/')[-1].split('_')[3].split('-')[1][-1]

        # Build fmri fname
        fmri_file = op.join(DATA_PATH, "shinobi.fmriprep", sub, ses, "func", f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        fmrifile_ok = op.isfile(fmri_file)
        if fmrifile_ok:
            nvol = nib.load(fmri_file).shape[-1]
            events = pd.read_csv(events_file, sep="\t")
            reps_df = events[events["trial_type"]=="gym-retro_game"]
            ntotreps = len(reps_df)
            for _, rep in reps_df.iterrows():
                if type(rep["stim_file"]) == str and not "Missing file" in rep["stim_file"]:
                    usable_reps += 1
                    if rep["level"] == "level-1":
                        nlvl1 += 1
                    elif rep["level"] == "level-4":
                        nlvl4 += 1
                    elif rep["level"] == "level-5":
                        nlvl5 += 1

                    json_fname = rep["stim_file"].replace(".bk2", ".json")
                    with open(op.join(DATA_PATH, "shinobi", json_fname)) as f:
                        sidecar = json.load(f)
                    nclear += int(sidecar["cleared"])
                    nhealthloss += int(sidecar["total health lost"])
                    nkill += int(sidecar["enemies killed"])
                    
            sub_list.append(sub)
            ses_list.append(ses)
            run_list.append(f"run-0{run}")
            fmrifile_list.append(fmri_file)
            nvol_list.append(nvol)
            nrep_total_list.append(ntotreps)
            nrep_usable_list.append(usable_reps)
            nlvl1_list.append(nlvl1)
            nlvl4_list.append(nlvl4)
            nlvl5_list.append(nlvl5)
            nclear_list.append(nclear)
            nhealthloss_list.append(nhealthloss)
            nkill_list.append(nkill)



    data_df = pd.DataFrame({
        "sub" : sub_list,
        "ses" : ses_list,
        "run" : run_list,
        "fmri_file" : fmrifile_list,
        "nvol" : nvol_list,
        "nrep_total" : nrep_total_list,
        "nrep_usable" : nrep_usable_list,
        "nlvl1" : nlvl1_list,
        "nlvl4" : nlvl4_list,
        "nlvl5" : nlvl5_list,
        "nclear" : nclear_list,
        "nhealthloss" : nhealthloss_list,
        "nkill" : nkill_list,

    })
    data_df.to_csv(dataset_fname, index=False)
    return data_df


dataset_fname = op.join(DATA_PATH, "processed", "shinobi_dataset_description.csv")

if not op.isfile(dataset_fname):
    data_df = build_dataset_description(dataset_fname)
else:
    data_df = pd.read_csv(dataset_fname)