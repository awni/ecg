#!/usr/bin/env python
"""
Export ECG and label data.

Usage:
 export.py  <db_file> <out_dir> (--train|--test|--val|--all)

Arguments:
<db_file>                    Path to pickle file containing episodes' info
<out_dir>                    Output directory to save the results

Options:
--train|--test|--val|--all   Whether to extract the data for either of the
                             training, testing, validation sets or the data
                             for all of the three sets together.
-h --help                    Show this documentation.
"""


import os
import numpy as np
import cPickle as pickle
from docopt import docopt

from db_constants import ECG_BYTES_PER_SAMPLE, EPI_EXT, ECG_EXT

qa = '_post'


def _extract(epi_info):
    """
    Extracts raw ecg using the input info.
    :param epi_info: dictionary with rec, start_idx, end_idx, and
                    rhythm_name keys, determining record path, starting index,
                    ending index and rhythm name, respectively.
    :return: raw ECG signal plus the rhythm name
    """
    start_idx = int(epi_info['start_idx'])
    start_byte = int(start_idx * ECG_BYTES_PER_SAMPLE)
    n_samples = int(epi_info['end_idx'] - epi_info['start_idx'])

    epi_path = epi_info['rec']
    rec_id = os.path.basename(epi_path).split(qa+EPI_EXT)[0]
    rec_dir = os.path.dirname(epi_path)

    ecg_path = os.path.join(rec_dir, rec_id+ECG_EXT)

    with open(ecg_path, 'rb') as ecg_file:
        ecg_file.seek(start_byte)
        ecg_sig = np.fromfile(ecg_file, np.int16, n_samples)

    return ecg_sig, epi_info['rhythm_name']


def _save(ecg_sig, label, out_dir, idx):
    """
    Saves ECG with numpy.save (load with numpy.load) and pickles labels.
    :param ecg_sig: ECG array
    :param label: rhythm label
    :param out_dir: output directory to save the results
    :param idx: output file name
    """
    np.save(os.path.join(out_dir, str(idx)), ecg_sig)
    with open(os.path.join(out_dir, "{}.pkl".format(idx)), 'w') as f:
        pickle.dump(label, f)

if __name__ == '__main__':
    args = docopt(__doc__)

    db_path = args['<db_file>']
    out_dir = args['<out_dir>']

    # check if output directory exists and make it if not
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args['--train']:
        export_set = 'train'
    elif args['--test']:
        export_set = 'test'
    elif args['--val']:
        export_set = 'val'
    else:
        export_set = 'all'

    with open(db_path, 'rb') as f:
        db_info = pickle.load(f)

    if export_set == 'all':
        set_info = db_info[~db_info.set.isnull()]
    else:
        set_info = db_info[db_info.set == export_set]

    for idx, epi_info in set_info.iterrows():
        ecg_sig, label = _extract(epi_info)
        _save(ecg_sig, label, out_dir, idx)

    print 'Data export completed!'





