from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import range
import json
import numpy as np
import fnmatch
import os
from tqdm import tqdm

# ECG constants
ECG_SAMP_RATE = 200.0  # Hz
ECG_EXT = '.ecg'
EPI_EXT = None
qa = '_post'


def _find_all_files(src, qa, ext):
    for root, dirnames, filenames in os.walk(src):
        for filename in fnmatch.filter(filenames, '*' + qa + ext):
            yield(os.path.join(root, filename))


def get_all_records(src):
    return _find_all_files(src, '', ECG_EXT)


def patient_id(record):
    return os.path.basename(record).split("_")[0]


def stratify(records, val_frac):

    blacklist = []

    def build_blacklist():
        for record in get_all_records('./data/label_review'):
            pid = patient_id(record)
            blacklist.append(pid)
        return blacklist

    def get_bucket_from_id(pat):
        return int(int(pat, 16) % 10)

    val, train = [], []
    for record in tqdm(records):
        pid = patient_id(record)
        if pid in blacklist:
            print(pid + ' in blacklist, skipping')
            continue
        bucket = get_bucket_from_id(pid)
        chosen = val if bucket < (val_frac * 10) else train
        chosen.append(record)
    return train, val


def round_to_step(n, step):
    diff = (n - 1) % step
    if diff < (step / 2):
        return n - diff
    else:
        return n + (step - diff)


def load_episodes(record, step):
    base = os.path.splitext(record)[0]
    ep_json = base + EPI_EXT
    with open(ep_json, 'r') as fid:
        episodes = json.load(fid)['episodes']

    # Round onset samples to the nearest step
    for episode in episodes:
        episode['onset_round'] = round_to_step(episode['onset'], step)

    # Set offset to onset - 1
    for e, episode in enumerate(episodes):
        # For the last episode set to the end
        if e == len(episodes) - 1:
            episode['offset_round'] = episode['offset']
        else:
            if(episodes[e+1]['onset_round'] != round_to_step(episode['offset'] + 1, step)):
                print('Something wrong with data in... ' + ep_json)
                # return None
            episode['offset_round'] = episodes[e+1]['onset_round'] - 1

    return episodes


def make_labels(episodes, duration, step):
    labels = []
    for episode in episodes:
        rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
        rhythm_labels = int(rhythm_len / step)
        rhythm = [episode['rhythm_name']] * rhythm_labels
        labels.extend(rhythm)
    dur_labels = int(duration * ECG_SAMP_RATE / step)
    labels = [labels[i:i+dur_labels]
              for i in range(0, len(labels) - dur_labels + 1, dur_labels)]
    return labels


def load_ecg(record, duration, step):
    with open(record, 'r') as fid:
        ecg = np.fromfile(fid, dtype=np.int16)

    n_per_win = int(duration * ECG_SAMP_RATE / step) * step

    # Truncate to a multiple of the duration
    ecg = ecg[:n_per_win * int(len(ecg) / n_per_win)]

    # Split into consecutive duration segments
    ecg = ecg.reshape((-1, n_per_win))
    n_segments = ecg.shape[0]
    segments = [arr.squeeze()
                for arr in np.vsplit(ecg, range(1, n_segments))]
    return segments


def construct_dataset(records, duration, step=ECG_SAMP_RATE):
    data = []
    for record in tqdm(records):
        episodes = load_episodes(record, step)
        if episodes is not None:
            labels = make_labels(episodes, duration, step)
            segments = load_ecg(record, duration, step)
            data.extend(zip(segments, labels))
    return data


def load_all_data(data_path, duration, val_frac, step=ECG_SAMP_RATE,
                  toy=False, extension='.episodes.json'):
    # _rev0.episodes
    global EPI_EXT
    EPI_EXT = extension
    print('Stratifying records...')
    #build_blacklist()
    train, val = stratify(get_all_records(data_path), val_frac=val_frac)
    if toy is True:
        print('Using toy dataset...')
        train = train[:1000]
        val = val[:100]
    print('Constructing Training Set...')
    train = construct_dataset(train, duration, step=step)
    print('Constructing Validation Set...')
    val = construct_dataset(val, duration, step=step)
    return train, val
