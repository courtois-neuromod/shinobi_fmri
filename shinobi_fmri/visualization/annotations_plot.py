import pandas as pd
from shinobi_fmri.annotations.annotations import trim_events_df, plot_gameevents
from shinobi_fmri.config import DATA_PATH as path_to_data, FIG_PATH as figures_path
import os
import matplotlib.pyplot as plt
import logging

def main():
    annot_path = path_to_data + 'processed/annotations/'
    fig_path = figures_path + 'annotations/'
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    files = os.listdir(annot_path)
    for file in files:
        events_df = pd.read_csv(annot_path + file)
        if not events_df.empty:
            trimmed_df = trim_events_df(events_df, trim_by='event')
            fig, ax = plot_gameevents(trimmed_df)
            plt.savefig(fig_path + file[:-3] + 'png', bbox_inches='tight')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
