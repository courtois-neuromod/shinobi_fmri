import shinobi_fmri.config as config
import seaborn as sbn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os.path as op
import ptitprince

def raincloud_fd(mean_fds_dataframe):
    sns.set(font_scale = 1.5)
    dx="subject"; dy="FD"; dhue = "subject"; ort="v"; pal = "Set2"; sigma = .15
    f, ax = plt.subplots(figsize=(10,10))
    ax=ptitprince.RainCloud(x = dx, y = dy, hue = dhue, data = mean_fds_dataframe, palette = pal, bw = sigma,
    width_viol = 1, ax = ax, orient = ort, alpha = .65, offset=-.05,move=.2,  width_box=.1)
    ax.set_title('Average framewise displacement (per run)', fontsize=24)
    ax.set_xlabel('Subject', fontsize=20)
    ax.set_ylabel('FD', fontsize=20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.legend(loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:4])
    return f, ax


figures_path = config.figures_path #'/home/hyruuk/GitHub/neuromod/shinobi_fmri/reports/figures/'
path_to_data = config.path_to_data  #'/media/storage/neuromod/shinobi_data/'
subjects = config.subjects
bounds_quantiles = [[0,0.01], [0.01,0.10], [0.10,0.50], [0.50,0.90], [0.90,0.99], [0,1]]
fds_dict = {'FD':[],
            'subject':[],
            'session':[],
            'run':[],
            'run_id':[]}
for sub in subjects:
    sessions = os.listdir(op.join(path_to_data, "shinobi", sub))
    for ses in sorted(sessions):
        print(f"Processing {sub} {ses}")
        runs = [
            filename[-12]
            for filename in os.listdir(
                op.join(path_to_data, "shinobi", sub, ses, "func")
            )
            if "events.tsv" in filename
        ]
        for run in sorted(runs):
            try:
                conf_fname = op.join(
                    path_to_data,
                    "shinobi",
                    "derivatives",
                    "fmriprep-20.2lts",
                    "fmriprep",
                    #'/lustre03/project/6003287/datasets/cneuromod_processed/fmriprep/shinobi/',
                    sub,
                    ses,
                    "func",
                    f"{sub}_{ses}_task-shinobi_run-{run}_desc-confounds_timeseries.tsv",
                )
                confounds = pd.read_csv(conf_fname, sep='\t')
                fd_timeseries = confounds['framewise_displacement'].dropna()
                for fd in fd_timeseries:
                    fds_dict['FD'].append(fd)
                    fds_dict['subject'].append(sub)
                    fds_dict['session'].append(ses)
                    fds_dict['run'].append(run)
                    fds_dict['run_id'].append(f'{sub}_{ses}_{run}')
            except Exception as e:
                print(e)
                print(f'Run {run} missing')

fds_dataframe = pd.DataFrame(fds_dict)
fds_dataframe_avg = fds_dataframe.groupby(by=['run_id', 'subject']).mean()
fds_dataframe_avg.reset_index(inplace=True)
fds_dataframe_avg = fds_dataframe_avg.rename(columns = {'index':'Items'})


f, ax = raincloud_fd(fds_dataframe_avg)
#sbn.boxenplot(data=mean_fds_dataframe, x='subject', y='FD', scale='area')
f.savefig(op.join(figures_path, 'frames_fds.png'))


# Generate mean FD plot
mean_fds_dict = copy.deepcopy(fds_dict)
mean_fds_dict['quantiles'] = ['' for x in range(len(fds_dict['FD']))]
for run_idx in range(len(fds_dict['FD'])):
    mean_fds_dict['FD'][run_idx] = np.mean(np.array(fds_dict['FD'][run_idx]))
    mean_fds_dict['quantiles'][run_idx] = '0-100'
mean_fds_dataframe = pd.DataFrame(mean_fds_dict)


f, ax = raincloud_fd(mean_fds_dataframe)
#sbn.boxenplot(data=mean_fds_dataframe, x='subject', y='FD', scale='area')
f.savefig(op.join(figures_path, 'mean_fds.png'))

mean_fds_dataframe.to_csv(op.join(path_to_data, 'processed', 'fds_dict.csv'))

# Plot per quantile
for bounds in bounds_quantiles:
    for run_idx in range(len(fds_dict['FD'])):
        lowbound = np.quantile(fds_dict['FD'][run_idx], bounds[0])
        highbound = np.quantile(fds_dict['FD'][run_idx], bounds[1])
        mean_fds_dict['FD'].append(np.mean([x for x in fds_dict['FD'][run_idx] if x > lowbound and x < highbound]))
        mean_fds_dict['subject'].append(fds_dict['subject'][run_idx])
        mean_fds_dict['session'].append(fds_dict['session'][run_idx])
        mean_fds_dict['run'].append(fds_dict['run'][run_idx])
        mean_fds_dict['quantiles'].append(f'{bounds[0]}-{bounds[1]}')

mean_fds_dataframe = pd.DataFrame(mean_fds_dict)
mean_fds_dataframe.to_csv(op.join(path_to_data, 'processed', 'fds_dict_quantiles.csv'))
for bounds in bounds_quantiles:
    data_df = mean_fds_dataframe.loc[mean_fds_dataframe['quantiles']==f'{bounds[0]}-{bounds[1]}']
    f, ax = raincloud_fd(data_df)
    #sbn.boxenplot(data=mean_fds_dataframe.loc[mean_fds_dataframe['quantiles']==f'{bounds[0]}-{bounds[1]}'], x='subject', y='FD', scale='area')
    f.savefig(op.join(figures_path, f'fds_dict_quantiles_{bounds[0]}-{bounds[1]}.png'))
    plt.close()


def get_outliers_table(fds_dict, save=True, threshold=0.5):
    outliers_dict = {'n_outliers':[],'n_frames':[],'outliers_perc':[],'subject':[],'session':[],'run':[]}
    for run_idx, run in enumerate(fds_dict['FD']):
        n_outliers = len([x for x in fds_dict['FD'][run_idx] if x > threshold])
        n_frames = len(fds_dict['FD'][run_idx])
        outliers_dict['n_outliers'].append(n_outliers)
        outliers_dict['n_frames'].append(n_frames)
        outliers_dict['outliers_perc'].append((n_outliers/n_frames)*100)
        outliers_dict['subject'].append(fds_dict['subject'][run_idx])
        outliers_dict['session'].append(fds_dict['session'][run_idx])
        outliers_dict['run'].append(fds_dict['run'][run_idx])
    if save:
        outliers_df = pd.DataFrame(outliers_dict)
        outliers_df_fpath = op.join(path_to_data, 'processed', f'fds_outliers_{threshold}.csv')
        outliers_df.to_csv(outliers_df_fpath)
        #sbn.boxenplot(data=outliers_df, y='outliers_perc', x='subject').set(title=f'% of frames with FD > {threshold}')

        plt.savefig(op.join(figures_path, f'fds_outliers_{threshold}.png'))
        plt.close()
    return outliers_dict


outliers_05 = get_outliers_table(fds_dict, threshold=0.5)
outliers_02 = get_outliers_table(fds_dict, threshold=0.2)
