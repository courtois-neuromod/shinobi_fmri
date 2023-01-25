import os.path as op

def get_filenames(sub, ses, run, path_to_data):
    fmri_fname = op.join(
        path_to_data,
        "shinobi.fmriprep",
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    assert op.isfile(fmri_fname), f"fMRI file not found for {sub}_{ses}_{run}"

    anat_fname = op.join(
        path_to_data,
        "cneuromod.processed",
        "smriprep",
        sub,
        "anat",
        f"{sub}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    assert op.isfile(anat_fname), f"sMRI file not found for {sub}_{ses}_{run}"

    events_fname = op.join(
        path_to_data, 
        "shinobi", 
        sub,
        ses,
        "func",
        f"{sub}_{ses}_task-shinobi_run-0{run}_annotated_events.tsv"
    )
    assert op.isfile(events_fname), f"Annotated events file not found for {sub}_{ses}_{run}" 

    glm_fname = op.join(path_to_data,
                    "processed25122022",
                    "glm",
                    "run-level",
                    f"{sub}_{ses}_run-0{run}_fitted_glm.pkl")

    os.makedirs(op.join(path_to_data,
                        "processed25122022",
                        "glm",
                        "run-level"), exist_ok=True)

    return fmri_fname, anat_fname, events_fname, glm_fname
