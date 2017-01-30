from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import zip
from builtins import int
from builtins import range
import collections
import json
import numpy as np
import os
from tqdm import tqdm

from .dataset_tools.db_constants import ECG_SAMP_RATE
from .dataset_tools.db_constants import ECG_EXT, EPI_EXT
from .dataset_tools.extract_episodes import _find_all_files, qa


def get_all_records(src):
    """
    Find all the ECG files.
    """
    return _find_all_files(src, '', ECG_EXT)


def stratify(records, val_frac):
    """
    Splits the data by patient into train and validation.

    Returns a tuple of two lists (train, val). Each list contains the
    corresponding records.
    """
    def patient_id(record):
        return os.path.basename(record).split("_")[0]

    patients = collections.defaultdict(list)
    for record in tqdm(records):
        patients[patient_id(record)].append(record)
    patients = sorted(list(patients.values()))
    np.random.shuffle(patients)
    cut = int(len(patients) * val_frac)
    train, val = patients[cut:], patients[:cut]
    train = [record for patient in train for record in patient]
    val = [record for patient in val for record in patient]
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
    """
    List of ecg records, duration to segment them into.
    :param duration: The length of examples in seconds.
    :param step: Number of samples to step the label by.
                 (e.g. step=200 means new label every second).
    """
    data = []
    for record in tqdm(records):
        episodes = load_episodes(record, step)
        labels = make_labels(episodes, duration, step)
        segments = load_ecg(record, duration, step)
        data.extend(zip(segments, labels))
    return data


def load_all_data(data_path, duration, val_frac, step=ECG_SAMP_RATE,
                  toy=False):
    print('Stratifying records...')
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

if __name__ == "__main__":
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
        msg = "Bad round: {} didn't round to {} ."
        assert round_to_step(n, 200) == m, msg.format(n, m)

    print("Tests passed!")
