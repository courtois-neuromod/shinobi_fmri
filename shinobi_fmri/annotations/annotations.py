import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shinobi_behav.features.features import filter_run, compute_framewise_aps
import matplotlib
import matplotlib.collections as mc
from shinobi_behav.params import actions


def generate_key_events(repvars, key, FS=60):
    """Create a Nilearn compatible events dataframe containing Health Loss events

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition
    key : string
        Name of the action variable to process
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in Nilearn-compatible format containing the
        corresponding action events.
    """
    var = repvars[key]
    # always keep the first and last value as 0 so diff will register the state transition
    var[0] = 0
    var[-1] = 0

    var_bin = [int(val) for val in var]
    diffs = list(np.diff(var_bin, n=1))
    presses = [round(i/FS, 3) for i, x in enumerate(diffs) if x == 1]
    releases = [round(i/FS, 3) for i, x in enumerate(diffs) if x == -1]
    onset = presses
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    trial_type = ['{}'.format(key) for i in range(len(presses))]
    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type})
    return events_df


def generate_aps_events(repvars, FS=60, min_dur=1):
    """Create a Nilearn compatible events dataframe containing Low and High APS
    events, based on a median split.

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition.
    FS : int
        The sampling rate of the .bk2 file
    min_dur : float
        Minimal duration of a Low or High APS segment, defaults to 1 (sec)

    Returns
    -------
    events_df :
        An events DataFrame in Nilearn-compatible format containing the
        Low and High APS events.
    """
    framewise_aps = compute_framewise_aps(repvars, actions=actions, FS=FS)
    filtered_aps = filter_run(framewise_aps, order=3, cutoff=0.002)
    var = filtered_aps

    median = np.median(var)

    mask_high = np.zeros(len(var))
    mask_low = np.zeros(len(var))

    for i, timestep in enumerate(var[1:-1]): # always keep the first and last value as 0 so diff will register the state transition
        if timestep < median:
            mask_low[i+1] = 1
        if timestep > median:
            mask_high[i+1] = 1

    diff_high = np.diff(mask_high, n=1)
    diff_low = np.diff(mask_low, n=1)

    durations_high = np.array([i for i, x in enumerate(diff_high) if x == -1]) - np.array([i for i, x in enumerate(diff_high) if x == 1])
    durations_low = np.array([i for i, x in enumerate(diff_low) if x == -1]) - np.array([i for i, x in enumerate(diff_low) if x == 1])

    #build df
    onset = []
    duration = []
    trial_type = []
    for i, dur in enumerate(durations_high):
        if dur >= (min_dur*FS):
            onset.append(np.array([i for i, x in enumerate(diff_high) if x == 1])[i]/FS)
            duration.append(durations_high[i]/FS)
            trial_type.append('high_APS')
    for i, dur in enumerate(durations_low):
        if dur >= (min_dur*FS):
            onset.append(np.array([i for i, x in enumerate(diff_low) if x == 1])[i]/FS)
            duration.append(durations_low[i]/FS)
            trial_type.append('low_APS')

    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type})
    return events_df

def generate_healthloss_events(repvars, FS=60, dur=0.1):
    """Create a Nilearn compatible events dataframe containing Health Loss events

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file
    dur : float
        Arbitrary duration of the generated event, defaults to 0.1

    Returns
    -------
    events_df :
        An events DataFrame in Nilearn-compatible format containing the
        Health Loss and Gain events.
    """
    health = repvars['health']
    diff_health = np.diff(health, n=1)

    onset = []
    duration = []
    trial_type = []
    for idx, x in enumerate(diff_health):
        if x < 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthLoss')
        if x > 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthGain')

    #build df
    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type})
    return events_df

def create_runevents(runvars, actions, FS=60, min_dur=1, get_aps=True, get_actions=True, get_healthloss=True, get_startend=True):
    """Create a Nilearn compatible events dataframe from game variables and start/duration info of repetitions

    Parameters
    ----------
    runvars : list
        A list of repvars dicts, corresponding to the different repetitions of a run
    actions : list of strings
        A list that contains the name of all the action variables
    FS : int
        The sampling rate of the .bk2 file
    min_dur : float
        Minimum duration of an event, currently only used for APS events
    get_aps : boolean
        If True, generates low/high APS (action per second) segments based on framewise_APS
    get_actions : boolean
        If True, generates actions events based on key presses
    get_healthloss : boolean
        If True, generates health loss events based on changes on the "lives" variable
    get_startend : boolean
        If True, generates events indicating the start and end of each repetition

    Returns
    -------
    events_df :
        An events DataFrame in Nilearn-compatible format.
    """

    # init df list
    all_df = []
    for idx, repvars in enumerate(runvars):
        if get_actions:
            for act in actions:
                temp_df = generate_key_events(repvars, act, FS=FS)
                temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
                temp_df['trial_type'] = repvars['level'] + '_' + temp_df['trial_type']
                all_df.append(temp_df)

        if get_aps:
            temp_df = generate_aps_events(repvars, FS=FS, min_dur=1)
            temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
            temp_df['trial_type'] = repvars['level'] + '_' + temp_df['trial_type']
            all_df.append(temp_df)

        if get_healthloss:
            temp_df = generate_healthloss_events(repvars, FS=FS, dur=0.1)
            temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
            temp_df['trial_type'] = repvars['level'] + '_' + temp_df['trial_type']
            all_df.append(temp_df)

        if get_startend:
            temp_df = pd.DataFrame([
                                     [repvars['rep_duration'],
                                     repvars['rep_onset'],
                                     'level_{}'.format(repvars['level'])]],
                                     columns=['duration', 'onset', 'trial_type'])
            all_df.append(temp_df)

        #todo : if get_endstart -- what did I mean by that ?
        #todo : if get_kills

    try:
        events_df = pd.concat(all_df).sort_values(by='onset').reset_index(drop=True)
    except ValueError:
        print('No bk2 files available for this run. Returning empty df.')
        events_df = pd.DataFrame()
    return events_df


def trim_events_df(events_df, trim_by='LvR'):
    """Creates a new events_df that contains only the conditions of interest.

    Parameters
    ----------
    events_df : DataFrame
        The original dataframe created with create_runevents, containing all the events
    trim_by : string
        A string that indicates which types of events are to be kept. Can be
        "LvR" (for Left vs Right hand events)
        "event" (all events, irrespective of level played)
        "healthloss" (health loss events)
        "JvH" (Jump vs Hit events)

    Returns
    -------
    trimmed_df :
        An events DataFrame in Nilearn-compatible format, containin only the conditions of interest
    """
    if trim_by=='LvR':
        # Create Left df
        lh_df = pd.concat([events_df[events_df['trial_type'] == '1-0_LEFT'],
                           events_df[events_df['trial_type'] == '1-0_RIGHT'],
                           events_df[events_df['trial_type'] == '1-0_DOWN'],
                           events_df[events_df['trial_type'] == '1-0_UP'],
                           events_df[events_df['trial_type'] == '4-1_LEFT'],
                           events_df[events_df['trial_type'] == '4-1_RIGHT'],
                           events_df[events_df['trial_type'] == '4-1_DOWN'],
                           events_df[events_df['trial_type'] == '4-1_UP'],
                            events_df[events_df['trial_type'] == '5-0_LEFT'],
                           events_df[events_df['trial_type'] == '5-0_RIGHT'],
                           events_df[events_df['trial_type'] == '5-0_DOWN'],
                           events_df[events_df['trial_type'] == '5-0_UP']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_df['trial_type'] = 'LeftH'

        # Create Right df
        rh_df = pd.concat([events_df[events_df['trial_type'] == '1-0_B'],
                           events_df[events_df['trial_type'] == '1-0_C'],
                           events_df[events_df['trial_type'] == '4-1_B'],
                           events_df[events_df['trial_type'] == '4-1_C'],
                            events_df[events_df['trial_type'] == '5-0_B'],
                           events_df[events_df['trial_type'] == '5-0_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_df['trial_type'] = 'RightH'
        # Regroup and pass them
        trimmed_df = pd.concat([lh_df, rh_df]).sort_values(by='onset').reset_index(drop=True)

    if trim_by=='event':
        # mostly for plotting
        lh_l = pd.concat([ events_df[events_df['trial_type'] == '1-0_LEFT'],
                           events_df[events_df['trial_type'] == '4-1_LEFT'],
                           events_df[events_df['trial_type'] == '5-0_LEFT']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_l['trial_type'] = 'Left hand - Move left'
        lh_r = pd.concat([ events_df[events_df['trial_type'] == '1-0_RIGHT'],
                           events_df[events_df['trial_type'] == '4-1_RIGHT'],
                           events_df[events_df['trial_type'] == '5-0_RIGHT']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_r['trial_type'] = 'Left hand - Move right'
        lh_d = pd.concat([ events_df[events_df['trial_type'] == '1-0_DOWN'],
                           events_df[events_df['trial_type'] == '4-1_DOWN'],
                           events_df[events_df['trial_type'] == '5-0_DOWN'],
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_d['trial_type'] = 'Left hand - Move down'
        lh_u = pd.concat([ events_df[events_df['trial_type'] == '1-0_UP'],
                           events_df[events_df['trial_type'] == '4-1_UP'],
                           events_df[events_df['trial_type'] == '5-0_UP']
                          ]).sort_values(by='onset').reset_index(drop=True)
        lh_u['trial_type'] = 'Left hand - Move up'
        rh_jump = pd.concat([events_df[events_df['trial_type'] == '1-0_B'],
                           events_df[events_df['trial_type'] == '4-1_B'],
                            events_df[events_df['trial_type'] == '5-0_B']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_jump['trial_type'] = 'Right hand - Jump'
        rh_hit = pd.concat([events_df[events_df['trial_type'] == '1-0_C'],
                           events_df[events_df['trial_type'] == '4-1_C'],
                           events_df[events_df['trial_type'] == '5-0_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_hit['trial_type'] = 'Right hand - Hit'
        hl = pd.concat([events_df[events_df['trial_type'] == '1-0_HealthLoss'],
                           events_df[events_df['trial_type'] == '4-1_HealthLoss'],
                           events_df[events_df['trial_type'] == '5-0_HealthLoss']
                          ]).sort_values(by='onset').reset_index(drop=True)
        hl['trial_type'] = 'Health loss'
        trimmed_df = pd.concat([lh_l, lh_r, lh_u, lh_d, rh_jump, rh_hit, hl]).sort_values(by='onset').reset_index(drop=True)

    if trim_by=='healthloss':
        hl = pd.concat([events_df[events_df['trial_type'] == '1-0_HealthLoss'],
                           events_df[events_df['trial_type'] == '4-1_HealthLoss'],
                           events_df[events_df['trial_type'] == '5-0_HealthLoss']
                          ]).sort_values(by='onset').reset_index(drop=True)
        hl['trial_type'] = 'HealthLoss'
        trimmed_df = hl

    if trim_by=='JvH':
        # Jump vs Hit df
        rh_jump = pd.concat([events_df[events_df['trial_type'] == '1-0_B'],
                           events_df[events_df['trial_type'] == '4-1_B'],
                            events_df[events_df['trial_type'] == '5-0_B']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_jump['trial_type'] = 'Jump'
        rh_hit = pd.concat([events_df[events_df['trial_type'] == '1-0_C'],
                           events_df[events_df['trial_type'] == '4-1_C'],
                           events_df[events_df['trial_type'] == '5-0_C']
                          ]).sort_values(by='onset').reset_index(drop=True)
        rh_hit['trial_type'] = 'Hit'
        trimmed_df = pd.concat([rh_jump, rh_hit]).sort_values(by='onset').reset_index(drop=True)

    return trimmed_df


def plot_gameevents(events_df, colors='rand'):
    """Generates a plot of an events_df, showing the occurence and duration of
    all the events across the run.

    Parameters
    ----------
    events_df : DataFrame
        The original dataframe created with create_runevents, containing all the events
    colors : string
        The colors to use for each event type. Can be "rand" for random colors,
        or "lvr" for plotting Left and Right hand events with variations of red/blue

    Returns
    -------
    fig : matplotlib Figure
        A matplotlib figure displaying all events across the run
    ax : matplotlib Axis
        The corresponding axis
    """

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=40)

    trial_types = sorted(list(events_df.trial_type.unique()))

    event_ends = []
    for i in range(len(events_df['onset'])):
        event_ends.append(events_df['onset'][i] + events_df['duration'][i])

    total_duration = max(event_ends)
    time_axis = np.linspace(0, total_duration, 10000)

    if colors=='lvr':
        cmap_right = matplotlib.cm.get_cmap('hot')
        cmap_left = matplotlib.cm.get_cmap('cool')
        col_bank = [(0,0,0),cmap_left(0.5),cmap_left(0.4),cmap_left(0.3),cmap_left(0.2),cmap_right(0.5),cmap_right(0.3)]
    elif colors =='rand': # Generate random colors
        col_bank = []
        for i in range(0, len(trial_types)):
            col_bank.append(tuple(np.random.choice(range(0, 10), size=3)/10))
    else:
         col_bank = colors
    masks = []
    colors_segs = []
    segs = []
    for line in events_df.T.iteritems():
        onset = line[1][1]
        dur = line[1][2]
        ttype = line[1][3]
        segs.append([(onset, trial_types.index(ttype)+1), (onset+dur, trial_types.index(ttype)+1)])
        mask = np.ma.masked_where((time_axis > onset) & (time_axis < onset+dur), time_axis)
        masks.append(mask)
        colors_segs.append(col_bank[trial_types.index(ttype)])

    # create figure
    lc = mc.LineCollection(segs, colors=colors_segs, linewidths=77)
    fig, ax = plt.subplots(figsize=(15,10))

    ax.add_collection(lc)
    ax.set_yticks(np.arange(len(trial_types))+1)
    ax.set_yticklabels(trial_types)
    ax.set_ylim((0.5,len(trial_types)+0.5))
    ax.set_xlim((0,max(time_axis)))
    ax.set_xlabel('Time (s)', fontsize=30)
    ax.margins(0.1)
    return fig, ax

################# DEPRECATED
def plot_bidsevents(merged_df):
    event_ends = []
    for i in range(len(merged_df['onset'])):
        event_ends.append(merged_df['onset'][i] + merged_df['duration'][i])

    total_duration = max(event_ends)
    time_axis = np.linspace(0, total_duration, 10000)


    dict_to_plot = {}
    for ev_type in merged_df['trial_type'].unique():
        dict_to_plot[ev_type] = np.zeros(len(time_axis))

    for idx, line in merged_df.iterrows():
        for i, timepoint in enumerate(time_axis):
            if timepoint >= line['onset'] and timepoint <= line['onset']+line['duration']:
                dict_to_plot[line['trial_type']][i] = 1

    fig = plt.plot()
    plt.figure(figsize=(15,10))
    for i, key in enumerate(dict_to_plot.keys()):
        x = time_axis
        y = dict_to_plot[key]*(len(dict_to_plot.keys())-i)
        # remove 0's from plot
        for idx_val in reversed(range(len(y))):
            if y[idx_val] == 0:
                x = np.delete(x, idx_val)
                y = np.delete(y, idx_val)
        plt.scatter(x, y, label=key, marker = ',')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    return fig
