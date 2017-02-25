from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from builtins import zip
from builtins import range
import argparse
import collections
import fnmatch
import json
import numpy as np
import os

import process

ECG_SAMP_RATE = 200

class WaveLoader(object):
    def __init__(self, data_path='', test_frac=0.2, duration=10, **kwargs):

        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        self.processor = process.Processor(use_one_hot_labels=True,
                                    normalizer="robust_scale")
        self.data_path = data_path
        self.test_frac = test_frac
        self.duration = duration
        self.load()
        (self.x_train, self.y_train, self.x_test, self.y_test) = \
            self.processor.process(self)

    def load(self):
        records = get_all_records(self.data_path)
        train, test = stratify(records, self.test_frac)
        train_x_y_pairs = construct_dataset(train, self.duration)
        test_x_y_pairs = construct_dataset(test, self.duration)

        self.x_train, self.y_train = zip(*train_x_y_pairs)
        self.x_test, self.y_test = zip(*test_x_y_pairs)

        # Setup labels
        label_counter = collections.Counter(l for labels in self.y_train for l in labels)
        self.classes = sorted([c for c, _ in label_counter.most_common()])
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}


    @property
    def output_dim(self):
        return len(self.classes)

def get_all_records(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, "*.ecg"):
            yield(os.path.join(root, filename))

def patient_id(record):
    return os.path.basename(record).split("_")[0]

def stratify(records, test_frac):
    def get_bucket_from_id(pat):
        return int(int(pat, 16) % 10)

    test, train = [], []
    for record in records:
        pid = patient_id(record)
        bucket = get_bucket_from_id(pid)
        in_test = bucket < int(test_frac * 10)
        chosen = test if in_test else train
        chosen.append(record)
    return train, test

def load_waves(record):
    base = os.path.splitext(record)[0]
    wv_json = base + ".waves.json"
    with open(wv_json, 'r') as fid:
        waves = json.load(fid)['waves']
    waves = sorted(waves, key=lambda x: x['onset'])
    return waves

def make_labels(waves, duration):
    labels = []
    for wave in waves:
        wave_len = wave['offset'] - wave['onset'] + 1
        wave_labels = [wave['wave_name']] * wave_len
        labels.extend(wave_labels)
    dur_labels = int(duration * ECG_SAMP_RATE)
    labels = [labels[i:i+dur_labels]
              for i in range(0, len(labels) - dur_labels + 1, dur_labels)]
    return labels

def load_ecg(record, duration):
    with open(record, 'r') as fid:
        ecg = np.fromfile(fid, dtype=np.int16)

    n_per_win = int(duration * ECG_SAMP_RATE)

    ecg = ecg[:n_per_win * int(len(ecg) / n_per_win)]

    ecg = ecg.reshape((-1, n_per_win))
    n_segments = ecg.shape[0]
    segments = [arr.squeeze()
                for arr in np.vsplit(ecg, range(1, n_segments))]
    return segments

def construct_dataset(records, duration):
    data = []
    for record in records:
        waves = load_waves(record)
        labels = make_labels(waves, duration)
        segments = load_ecg(record, duration)
        data.extend(zip(segments, labels))
    return data

def load_train(params):
    loader = WaveLoader(**params)
    print("Length of training set {}".format(len(loader.x_train)))
    print("Length of test set {}".format(len(loader.x_test)))
    print("Output dimension {}".format(loader.output_dim))
    return loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    load_train(params)
