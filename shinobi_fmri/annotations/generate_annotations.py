from shinobi_behav.params import actions, path_to_data, subjects
from shinobi_behav.data.data import retrieve_scanvariables, extract_variables
from shinobi_behav.annotations.annotations import create_runevents
import pickle
import os
import logging
import pandas as pd


def main():
    # Create ann
    if not os.path.isdir(path_to_data + 'processed/annotations'):
        os.makedirs(path_to_data + 'processed/annotations')

    for sub in subjects:
        sessions = os.listdir(path_to_data + 'shinobi/' + sub)
        for ses in sorted(sessions):
            runs = [filename[-13] for filename in os.listdir(path_to_data + 'shinobi/' + '{}/{}/func'.format(sub, ses)) if 'bold.nii.gz' in filename] # change here to events.tsv
            for run in sorted(runs):
                events_fname = path_to_data + 'shinobi/{}/{}/func/{}_{}_task-shinobi_run-0{}_events.tsv'.format(sub, ses, sub, ses, run)
                eventsdf_path = path_to_data + 'processed/annotations/{}_{}_run-0{}.csv'.format(sub, ses, run)
                if not os.path.exists(eventsdf_path):
                    startevents = pd.read_table(events_fname)
                    files = startevents['stim_file'].values.tolist()
                    # Retrieve variables from these files
                    runvars = []
                    for idx, file in enumerate(files):
                        if isinstance(file, str):
                            filepath = path_to_data + 'shinobi/' + file
                            repvars = extract_variables(filepath)
                            repvars['rep_onset'] = startevents['onset'][idx]
                            repvars['rep_duration'] = startevents['duration'][idx]
                            runvars.append(repvars)
                    events_df = create_runevents(runvars, startevents, actions=actions)
                    events_df.to_csv(eventsdf_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
