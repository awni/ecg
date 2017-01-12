from __future__ import print_function
from __future__ import division

import collections
import glob
import json
import numpy as np
import os
import random

from dataset_tools.db_constants import ECG_SAMP_RATE
from dataset_tools.db_constants import ECG_EXT, EPI_EXT
from dataset_tools.extract_episodes import _find_all_files, qa

def get_all_records(src):
    """
    Find all the ECG files.
    """
    return list(_find_all_files(src, '', ECG_EXT))

def stratify(records, val_frac):
    """
    Splits the data by patient into train and validation.

    Returns a tuple of two lists (train, val). Each list contains the
    corresponding records.
    """
    def patient_id(record):
        return os.path.basename(record).split("_")[0]

    patients = collections.defaultdict(list)
    for record in records:
        patients[patient_id(record)].append(record)
    patients = patients.values()
    random.shuffle(patients)
    cut = int(len(patients) * val_frac)
    train, val = patients[cut:], patients[:cut]
    train = [record for patient in train for record in patient]
    val = [record for patient in val for record in patient]
    return train, val

def round_to_second(n):
    rate = int(ECG_SAMP_RATE)
    diff = (n - 1) % rate
    if diff < (rate / 2):
        return n - diff
    else:
        return n + (rate - diff)

def load_episodes(record):
    base = os.path.splitext(record)[0]
    ep_json = base + EPI_EXT
    with open(ep_json, 'r') as fid:
        episodes = json.load(fid)['episodes']

    # Round onset samples to the nearest second
    for episode in episodes:
        episode['onset_round'] = round_to_second(episode['onset'])

    # Set offset to onset - 1
    for e, episode in enumerate(episodes):
        # For the last episode set to the end
        if e == len(episodes) - 1:
            episode['offset_round'] = episode['offset']
        else:
            episode['offset_round'] = episodes[e+1]['onset_round'] - 1

    return episodes

def make_labels(episodes, duration):
    labels = []
    for episode in episodes:
        rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
        rhythm_secs = int(rhythm_len / ECG_SAMP_RATE)
        rhythm = [episode['rhythm_name']] * rhythm_secs
        labels.extend(rhythm)
    labels = [labels[i:i+duration]
               for i in range(0, len(labels) - duration + 1, duration)]
    return labels

def load_ecg(record, duration):
    with open(record, 'r') as fid:
        ecg = np.fromfile(fid, dtype=np.int16)

    n_per_win = int(duration * ECG_SAMP_RATE)

    # Truncate to a multiple of the duration
    ecg = ecg[:n_per_win * int(len(ecg) / n_per_win)]

    # Split into consecutive duration segments
    ecg = ecg.reshape((-1, n_per_win))
    n_segments = ecg.shape[0]
    segments = [arr.squeeze()
                 for arr in np.vsplit(ecg, range(1, n_segments))]
    return segments

def construct_dataset(records, duration):
    """
    List of ecg records, duration to segment them into.
    """
    data = []
    for record in records:
        episodes = load_episodes(record)
        labels = make_labels(episodes, duration)
        segments = load_ecg(record, duration)
        data.extend(zip(segments, labels))
    return data

def load_all_data(data_path, duration, val_frac):
    all_records = get_all_records(data_path)
    train, val = stratify(all_records, val_frac=val_frac)
    train = construct_dataset(train, duration)
    val = construct_dataset(val, duration)
    return train, val

if __name__ == "__main__":
    random.seed(2016)

    src = "/deep/group/med/irhythm/ecg/clean_30sec_recs/batch1"
    duration = 30
    val_frac = 0.1

    import time
    start = time.time()
    train, val = load_all_data(src, duration, val_frac)
    print("Training examples: {}".format(len(train)))
    print("Validation examples: {}".format(len(val)))
    print("Load time: {:.3f} (s)".format(time.time() - start))

    # Some tests
    for n, m in [(401, 401), (1, 1), (7, 1), (199, 201),
                 (200, 201), (101, 201), (100, 1)]:
        msg  = "Bad round: {} didn't round to {} ."
        assert round_to_second(n) == m, msg.format(n, m)

    print("Tests passed!")
