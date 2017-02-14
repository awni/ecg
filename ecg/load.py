from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from builtins import zip
from builtins import range
import json
import numpy as np
import fnmatch
import os
import warnings
import argparse
import collections
from tqdm import tqdm
from process import Processor

# FIXME: step and samp_rate and duration should be part of process, not load


class Loader(object):
    def __init__(
            self,
            data_path,
            processor,
            ecg_samp_rate=200.0,
            ecg_ext='.ecg',
            epi_ext='.episodes.json',
            blacklist_path="",
            duration=30,
            test_frac=0.2,
            step=200,
            toy=False,
            fit_processor=True,
            **kwargs):
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.TOY_LIMIT = 2000
        self.blacklist = []

        self.data_path = data_path
        self.ecg_samp_rate = ecg_samp_rate
        self.ecg_ext = ecg_ext
        self.epi_ext = epi_ext
        self.blacklist_path = blacklist_path
        self.duration = duration
        self.test_frac = test_frac
        self.step = step
        self.toy = toy
        self.processor = processor
        self.fit_processor = fit_processor

        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        self.load()
        self.setup_label_mappings()
        (self.x_train, self.y_train, self.x_test, self.y_test) = \
            self.processor.process(self, fit=self.fit_processor)

    def setup_label_mappings(self):
        y_tot = list(self.y_train) + list(self.y_test)
        label_counter = collections.Counter(
            l for labels in y_tot for l in labels)
        self.classes = sorted([c for c, _ in label_counter.most_common()])

        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def get_all_records(self, path):
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*' + self.ecg_ext):
                yield(os.path.join(root, filename))

    def patient_id(self, record):
        return os.path.basename(record).split("_")[0]

    def build_blacklist(self):
        print('Building blacklist...')
        self.blacklist = []
        for record in tqdm(self.get_all_records(self.blacklist_path)):
            pid = self.patient_id(record)
            self.blacklist.append(pid)

    def stratify(self, records):
        def get_bucket_from_id(pat):
            return int(int(pat, 16) % 10)

        test, train = [], []
        for record in tqdm(records):
            pid = self.patient_id(record)
            if len(self.blacklist) > 0 and pid in self.blacklist:
                continue
            bucket = get_bucket_from_id(pid)
            chosen = test if bucket < (self.test_frac * 10) else train
            chosen.append(record)
        return train, test

    def load_episodes(self, record):
        def round_to_step(n, step):
            diff = (n - 1) % step
            if diff < (step / 2):
                return n - diff
            else:
                return n + (step - diff)

        base = os.path.splitext(record)[0]
        ep_json = base + self.epi_ext
        with open(ep_json, 'r') as fid:
            episodes = json.load(fid)['episodes']
        episodes = sorted(episodes, key = lambda x: x['onset'])

        for episode in episodes:
            episode['onset_round'] = round_to_step(episode['onset'], self.step)

        for e, episode in enumerate(episodes):
            if e == len(episodes) - 1:
                episode['offset_round'] = episode['offset']
            else:
                if(episodes[e+1]['onset_round'] !=
                   round_to_step(episode['offset'] + 1, self.step)):
                    warnings.warn('Something wrong with data in... ' + ep_json,
                                  DeprecationWarning)
                episode['offset_round'] = episodes[e+1]['onset_round'] - 1

        return episodes

    def make_labels(self, episodes):
        labels = []
        for episode in episodes:
            rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
            rhythm_labels = int(rhythm_len / self.step)
            rhythm = [episode['rhythm_name']] * rhythm_labels
            labels.extend(rhythm)
        dur_labels = int(self.duration * self.ecg_samp_rate / self.step)
        labels = [labels[i:i+dur_labels]
                  for i in range(0, len(labels) - dur_labels + 1, dur_labels)]
        return labels

    def load_ecg(self, record):
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

        n_per_win = int(self.duration * self.ecg_samp_rate /
                        self.step) * self.step

        ecg = ecg[:n_per_win * int(len(ecg) / n_per_win)]

        ecg = ecg.reshape((-1, n_per_win))
        n_segments = ecg.shape[0]
        segments = [arr.squeeze()
                    for arr in np.vsplit(ecg, range(1, n_segments))]
        return segments

    def construct_dataset(self, records):
        data = []
        for record in tqdm(records):
            episodes = self.load_episodes(record)
            labels = self.make_labels(episodes)
            segments = self.load_ecg(record)
            if len(labels) == 0:
                print(episodes)
                print(labels)
            data.extend(zip(segments, labels))
        return data

    def load(self):
        if (self.blacklist_path != ""):
            self.build_blacklist()
        records = self.get_all_records(self.data_path)
        train, test = self.stratify(records)
        if self.toy is True:
            print('Using toy dataset...')
            train = train[:self.TOY_LIMIT]
            test = test[:self.TOY_LIMIT]
        print('Constructing Training Set...')
        train_x_y_pairs = self.construct_dataset(train)
        print('Constructing Test Set...')
        test_x_y_pairs = self.construct_dataset(test)

        if (len(train_x_y_pairs) > 0):
            self.x_train, self.y_train = zip(*train_x_y_pairs)
        else:
            self.x_train = self.y_train = []

        if (len(test_x_y_pairs) > 0):
            self.x_test, self.y_test = zip(*test_x_y_pairs)
        else:
            self.x_test = self.y_test = []

    @property
    def output_dim(self):
        return len(self.int_to_class)


def load_train(args, params):
    processor = Processor(**params)
    loader = Loader(args.data_path, processor, **params)

    print("Length of training set {}".format(len(loader.x_train)))
    print("Length of test set {}".format(len(loader.x_test)))
    print("Output dimension {}".format(loader.output_dim))
    return loader


def load_test(params_train, params_test):
    params_train["fit_processor"] = True
    params_test["fit_processor"] = False

    processor = Processor(**params_train)
    print("Fitting processor...")
    Loader(params_train["TRAIN_DATA_PATH"], processor, **params_train)
    print("Loading test dataset...")
    loader = Loader(params_test["EVAL_PATH"], processor, **params_test)

    print("Length of training set {}".format(len(loader.x_train)))
    print("Length of test set {}".format(len(loader.x_test)))
    print("Output dimension {}".format(loader.output_dim))
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    load_train(args, params)
