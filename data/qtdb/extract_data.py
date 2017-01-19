from __future__ import print_function

import glob
import numpy as np
import os
import random
import cPickle as pickle

DATA = "/deep/group/med/qtdb"
BASELINE = "BL"

# TODO, awni, figure out what these labels actually mean. They seem to
# correspond to r-peaks with slightly different morphology than for what the
# label N is given, or potentially r-peaks for a different rhythm.
REMAP = {"A" : "N", "B" : "N", "Q" : "N"}

def get_all_records():
    ecg_files = glob.glob(os.path.join(DATA, "*.npy"))
    return [os.path.splitext(ecg_file)[0] for ecg_file in ecg_files]

def stratify(records, val_frac=0.1):
    """
    Split into validation and train by record.
    :param records: List of records.
    :param val_frac: Fraction of number of records
                     to put in validation set.
    :return: (train, validation) lists of records.
    """
    random.shuffle(records)
    cut = int(val_frac * len(records))
    return records[cut:], records[:cut]

def load_ecg(record):
    """
    Reads and returns just the first channel of the raw ECG.
    """
    ecg = np.load(record + ".npy")
    return ecg[:, 1]

def load_labels(record):
    with open(record + ".pkl", 'r') as fid:
        labels = pickle.load(fid)
    waves = []
    wave = {}
    for l in labels:
        sample_n = l[1]
        w_l = l[2]
        if w_l == '(':
            wave['onset'] = sample_n
        elif w_l == ')':
            wave['offset'] = sample_n
            waves.append(wave)
            wave = {}
        else:
            # If we haven't seen a '(' yet then set onset to
            # the previous waves offset. We can have cases where
            # the wave boundary is blurred e.g. (p) (QRS) t)
            if 'onset' not in wave:
                wave['onset'] = waves[-1]['offset']
            wave['label'] = REMAP.get(w_l, w_l)
    return waves

def split_waves(waves, split_thresh=200):
    """
    Split wave labels when there is a large threshold in between
    the end of the previous wave and the start of the next. This
    usually signifies an unlabelled portion of the ECG stream.
    """
    splits = [0]
    for e, (w1, w2) in enumerate(zip(waves[:-1], waves[1:])):
        if w2['onset'] - w1['offset'] > split_thresh:
            splits.append(e + 1)
    splits.append(len(waves))
    waves = [waves[splits[i]:splits[i+1]]
              for i in range(len(splits[:-1]))]
    return waves

def make_examples(ecg, waves):
    examples = []
    for wave_set in waves:
        start = wave_set[0]['onset']
        stop = wave_set[-1]['offset']
        data = ecg[start:stop]
        labels = []
        for e, wv in enumerate(wave_set):
            labels.extend([wv['label']] * (wv['offset'] - wv['onset']))
            # Add baseline label in-between waves
            if e < len(wave_set) - 1:
                labels.extend([BASELINE] * (wave_set[e+1]['onset'] - wv['offset']))
        examples.append((data, labels))
    return examples

def load_all_records(records):
    all_examples = []
    for record in records:
        ecg = load_ecg(record)
        waves = split_waves(load_labels(record))
        examples = make_examples(ecg, waves)
        all_examples.extend(examples)
    return all_examples

def load_all_examples():
    records = get_all_records()
    train, val = stratify(records)
    train = load_all_records(train)
    val = load_all_records(val)
    return train, val

if __name__ == "__main__":
    random.seed(2017)
    train, val = load_all_examples()
    for data, labels in train:
        assert len(data) == len(labels), "Length mismatch."
    for data, labels in val:
        assert len(data) == len(labels), "Length mismatch."
    # Print some stats about the data.
    lengths = [len(data) for data, _ in train]
    print("Min length", min(lengths))
    print("Max length", max(lengths))
    print("Mean length", np.mean(lengths))
    labels = set(l for _, label in train for l in label)
    print(labels)

