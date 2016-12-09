#!/usr/bin/env python
"""
Identifies location of arbitrary-sized episodes within records of a given
data set.

Usage:
 extract_episodes.py  <epi_dir> --win=<win_length> --pause=<pause_length>
                      [--max_xt=<max_xt_per_rhythm>]

Arguments:
<epi_dir>                    Path to the root directory containing
                             episodes.json files (all the subdirectories
                             within the root will be searched)

Options:
--win=<win_length>           Window length (in seconds) for the extracted
                             episodes
--pause=<pause_length>       Minimum Pause length (in seconds) to be included
                             in the set
--max_xt=<max_xt_per_rhythm> Maximum number of extractions per rhythm-type per
                             record [default: 10]
-h --help                    Show this documentation.
"""


import os
import fnmatch
import pandas as pd
import cPickle as pickle
import json
from itertools import islice
from docopt import docopt


from db_constants import ECG_SAMP_RATE, EPI_EXT


qa = '_post'


def _find_all_files(src, qa, ext):
    """
    Finds all files satisfying the input criteria within the input directory,
    including its subdirectories.
    :param src: root directory to search for files
    :param qa: either '_pre', '_post', or '' determining file suffixes
    :param ext: file extensions
    """
    for root, dirnames, filenames in os.walk(src):
        for filename in fnmatch.filter(filenames, '*'+qa+ext):
            yield(os.path.join(root, filename))


def _extract_win_from_episode(rhy_epi, win_dur_samples, num_xt, rhy_name, rec):
    """
    :param rhy_epi: dictionary with onset, offset and dur_samples keys
                    identifying boundaries of an episode
    :param win_dur_samples: duration of windows (in samples) to be extracted
    :param num_xt: max number of windows to be extracted from this episode
    :param rhy_name: rhythm name for this episode
    :param rec: record name that this episode belongs to
    :return: a dataframe containing starting indices for the windows to be
             extracted from this episode.
    """
    # output initialization
    epi_info = pd.DataFrame(columns=['rec', 'rhythm_name', 'start_idx',
                                     'end_idx'])

    # starting indices of windows to be extracted
    # note: extracted windows have a len of win_dur_samples and are separated
    # from each other by a distance of win_dur_samples as well.
    win_idx = xrange(rhy_epi['onset'], rhy_epi['offset']-win_dur_samples,
                     win_dur_samples*2)
    win_idx_list = list(islice(win_idx, num_xt))

    epi_info['start_idx'] = win_idx_list
    epi_info = epi_info.assign(end_idx=epi_info.start_idx+win_dur_samples)
    epi_info['rec'] = rec
    epi_info['rhythm_name'] = rhy_name

    return epi_info

if __name__ == '__main__':
    args = docopt(__doc__)

    src = args['<epi_dir>']
    win_dur = int(args['--win'])  # sec
    min_pause_dur = int(args['--pause'])  # sec
    max_xt_per_rhythm = int(args['--max_xt'])

    win_dur_samples = int(win_dur * ECG_SAMP_RATE)
    half_win_dur_samples = win_dur_samples//2
    min_pause_dur_samples = int(min_pause_dur * ECG_SAMP_RATE)

    # output initialization
    recs_win_info = pd.DataFrame(columns=['rec', 'rhythm_name', 'start_idx',
                                          'end_idx'])

    recs = _find_all_files(src, qa, EPI_EXT)

    for rec in recs:
        with open(rec, 'r') as f:
            ee = json.load(f)

        epis = pd.DataFrame(ee['episodes'])
        rec_dur_samples = epis.offset.iloc[-1]  # total record length

        # select long enough episodes (treat PAUSE differently)
        epis = epis.assign(dur_samples=epis.offset-epis.onset+1)
        non_pause_epis = epis[(epis.dur_samples >= win_dur_samples) &
                              (epis.rhythm_name != 'PAUSE')]

        pause_epis = epis[(epis.rhythm_name == 'PAUSE') &
                          (epis.dur_samples >= min_pause_dur_samples) &
                          (epis.dur_samples < win_dur_samples)]

        # 1) extract windows from pause episodes
        for idx, rhy_epi in pause_epis.iloc[:max_xt_per_rhythm].iterrows():
            win_center_idx = (rhy_epi['onset'] + rhy_epi['offset'])//2
            win_onset = win_center_idx - half_win_dur_samples
            win_offset = win_dur_samples - half_win_dur_samples + win_center_idx

            if (win_onset < 0) or (win_offset > rec_dur_samples):
                continue

            recs_win_info = recs_win_info.append({'rec': rec,
                                                  'rhythm_name': 'PAUSE',
                                                  'start_idx': win_onset,
                                                  'end_idx': win_offset},
                                                 ignore_index=True)

        # 2) extract windows from non-pause episodes
        non_pause_epi_groups = non_pause_epis.groupby('rhythm_name')
        for rhy_name,  rhy_group in non_pause_epi_groups:
            # keep track of number of extractions per rhythm-type
            max_xt = max_xt_per_rhythm
            # select random intervals from each episode
            for idx, rhy_epi in rhy_group.iterrows():

                # select starting of extracted windows and update remaining
                # number of windows that can be extracted
                rec_rhy_windows = \
                    _extract_win_from_episode(rhy_epi, win_dur_samples, max_xt,
                                              rhy_name, rec)
                # update the output and the remaining number of extractions
                recs_win_info = recs_win_info.append(rec_rhy_windows,
                                                     ignore_index=True)
                max_xt -= len(rec_rhy_windows)

                if max_xt == 0:
                    # reached max extraction per rhythm-type per record;
                    # proceed to the next rhythm-type in that record
                    break

    # save the extracted windows info
    out_file = os.path.join(src, 'extracted_windows_df.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(recs_win_info, f)

    print 'Data extraction completed!'
    print "Episodes' info written in %s." % out_file













