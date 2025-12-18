import pandas as pd
import numpy as np
import os
import glob
import os.path as op
from shinobi_fmri.config import DATA_PATH, FIG_PATH
import nibabel as nib
import json
import matplotlib.pyplot as plt
import retro
from datetime import datetime
from itertools import product
import dataframe_image as dfi
from bids_loader.stimuli.game import get_variables_from_replay
import logging

def create_info_dict(repvars):
    info_dict = {}

    info_dict["duration"] = len(repvars["X_player"])/60

    lives_lost = sum([x for x in np.diff(repvars["lives"], n=1) if x < 0])
    if lives_lost == 0:
        cleared = True
    else:
        cleared = False
    info_dict["cleared"] = cleared

    info_dict["end_score"] = repvars["score"][-1]

    diff_health = np.diff(repvars["health"], n=1)
    try:
        index_health_loss = list(np.unique(diff_health, return_counts=True)[0]).index(-1)
        total_health_loss = np.unique(diff_health, return_counts=True)[1][index_health_loss]
    except Exception as e:
        print(e)
        total_health_loss = 0
    info_dict["total health lost"] = total_health_loss

    diff_shurikens = np.diff(repvars["shurikens"], n=1)
    try:
        index_shurikens_loss = list(np.unique(diff_shurikens, return_counts=True)[0]).index(-1)
        total_shurikens_loss = np.unique(diff_shurikens, return_counts=True)[1][index_shurikens_loss]
    except Exception as e:
        total_shurikens_loss = 0
    info_dict["shurikens used"] = total_shurikens_loss

    info_dict["enemies killed"] = len(generate_kill_events(repvars, FS=60, dur=0.1))
    return info_dict

def fix_position_resets(X_player):
    """Sometimes X_player resets to 0 but the player's position should keep
    increasing.
    This fixes it and makes sure that X_player is continuous. If not, the
    values after the jump are corrected.

    Parameters
    ----------
    X_player : list
        List of raw positions at each timeframe from one repetition.

    Returns
    -------
    list
        List of lists of fixed (continuous) positions. One per repetition.

    """

    fixed_X_player = []
    raw_X_player = X_player
    fix = 0  # keeps trace of the shift
    fixed_X_player.append(raw_X_player[0])  # add first frame
    for i in range(1, len(raw_X_player) - 1):  # ignore first and last frames
        if raw_X_player[i - 1] - raw_X_player[i] > 100:
            fix += raw_X_player[i - 1] - raw_X_player[i]
        fixed_X_player.append(raw_X_player[i] + fix)
    fixed_X_player.append(fixed_X_player[-1]) # re-add the last frame for consistency
    return fixed_X_player

def generate_kill_events(repvars, FS=60, dur=0.1):
    """Create a BIDS compatible events dataframe containing kill events,
    based on a sudden increase of score.
    + 200 pts : basic enemies (all levels)
    + 300 pts : mortars and machineguns (lvl 5)
    + 400 pts : cauldron-heads (level 1)
    + 500 pts : anti-riot cop (lvl 5) ; hovering ninja (lvl 4)


    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition.
    FS : int
        The sampling rate of the .bk2 file
    min_dur : float
        Minimal duration of a kill segment, defaults to 1 (sec)

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        kill events.
    """
    instant_score = repvars['instantScore']
    diff_score = np.diff(instant_score, n=1)

    onset = []
    duration = []
    trial_type = []
    level = []
    for idx, x in enumerate(diff_score):
        if x in [200,300]:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('Kill')
            level.append(repvars["level"])

    #build df
    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type,
                               'level':level})
    return events_df
def main():
    replayfile_list = sorted(glob.glob(op.join(DATA_PATH, "shinobi_training", "*", "*", "*", "*.bk2")))
    for replayfile in replayfile_list:
        print(replayfile)
        game = "ShinobiIIIReturnOfTheNinjaMaster-Genesis"
        repvars = get_variables_from_replay(replayfile, skip_first_step=replayfile, save_gif=True, game=game, inttype=retro.data.Integrations.STABLE)
        repvars["X_player"] = fix_position_resets(repvars["X_player"])
        # create json sidecar
        info_dict = create_info_dict(repvars)
        with open(replayfile.replace(".bk2", ".json"), "w") as outfile:
            json.dump(info_dict, outfile, default=str)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
