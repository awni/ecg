from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import range
import json
import numpy as np
import fnmatch
import os
import warnings
from tqdm import tqdm


def Loader(object):
    def __init__(
        self,
        data_path,
        ecg_samp_rate=200.0,
        ecg_ext='.ecg',
        epi_ext='.episodes.json',
        blacklist_path='./data/label_review',
        duration=30,
        val_frac=0.2,
        step=200,
        toy=False,
        **kwargs
    ):
        self.data_path = data_path
        self.ecg_samp_rate = ecg_samp_rate
        self.ecg_ext = ecg_ext
        self.epi_ext = epi_ext
        self.blacklist_path = blacklist_path
        self.duration = duration
        self.val_frac = val_frac
        self.step = step
        self.toy = toy
        self.TOY_LIMIT = 1000
        self.blacklist = []

        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

    def get_all_records(self):
        for root, dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, '*' + self.ecg_ext):
                yield(os.path.join(root, filename))

    def patient_id(self, record):
        return os.path.basename(record).split("_")[0]

    def build_blacklist(self):
        self.blacklist = []
        for record in get_all_records(self.blacklist_path):
            pid = patient_id(record)
            self.blacklist.append(pid)

    def stratify(self, records):
        def get_bucket_from_id(pat):
            return int(int(pat, 16) % 10)

        val, train = [], []
        for record in tqdm(records):
            pid = patient_id(record)
            if len(self.blacklist) > 0 and pid in self.blacklist:
                print(pid + ' in blacklist, skipping')
                continue
            bucket = get_bucket_from_id(pid)
            chosen = val if bucket < (self.val_frac * 10) else train
            chosen.append(record)
        return train, val

    def round_to_step(self, n, step):
        diff = (n - 1) % step
        if diff < (step / 2):
            return n - diff
        else:
            return n + (step - diff)

    def load_episodes(self, record, step):
        base = os.path.splitext(record)[0]
        ep_json = base + self.epi_ext
        with open(ep_json, 'r') as fid:
            episodes = json.load(fid)['episodes']

        for episode in episodes:
            episode['onset_round'] = round_to_step(episode['onset'], step)

        for e, episode in enumerate(episodes):
            if e == len(episodes) - 1:
                episode['offset_round'] = episode['offset']
            else:
                if(episodes[e+1]['onset_round'] !=
                   round_to_step(episode['offset'] + 1, step)):
                    warnings.warn('Something wrong with data in... ' + ep_json)
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
            if episodes is not None:
                labels = self.make_labels(episodes)
                segments = self.load_ecg(record)
                data.extend(zip(segments, labels))
        return data

    def load(self):
        if (self.blacklist_path is not None):
            print('Building blacklist...')
            self.build_blacklist()
        records = self.get_all_records()
        train, val = self.stratify(records)
        if self.toy is True:
            print('Using toy dataset...')
            train = train[:self.TOY_LIMIT]
            val = val[:self.TOY_LIMIT]
        print('Constructing Training Set...')
        train = self.construct_dataset(train)
        print('Constructing Validation Set...')
        val = self.construct_dataset(val)
        return train, val
