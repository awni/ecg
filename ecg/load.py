from __future__ import division
from __future__ import print_function
from builtins import dict
from builtins import zip
from builtins import range
import argparse
import collections
import numpy as np
import os
import random
import json
import joblib

import featurize
from data.irhythm.extract_data import load_all_data


class Loader(object):
    def __init__(
            self,
            data_path,
            batch_size=32,
            duration=30,
            step=200,
            val_frac=0.1,
            seed=None,
            use_one_hot_labels=True,
            use_cached_if_available=True,
            normalizer='min_max',
            wavelet_fns=[]):

        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        self.batch_size = batch_size
        self.duration = duration
        self.val_frac = val_frac
        self.wavelet_fns = wavelet_fns
        self.normalizer = normalizer
        self.use_one_hot_labels = use_one_hot_labels
        self.step = step

        self._load(data_path, use_cached_if_available)
        self._postprocess()

    def _postprocess(self):
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)

        if len(self.wavelet_fns) != 0:
            wavelet_transformer = \
                featurize.WaveletTransformer(self.wavelet_fns)
            self.x_train = wavelet_transformer.transform(self.x_train)
            self.x_test = wavelet_transformer.transform(self.x_test)

        if self.normalizer is not False:
            n = featurize.Normalizer(self.normalizer)
            n.fit(self.x_train)
            self.x_train = n.transform(self.x_train)
            self.x_test = n.transform(self.x_test)

        label_counter = collections.Counter(l for labels in self.y_train
                                            for l in labels)
        self.classes = sorted(
            [c for c, _ in label_counter.most_common()])  # FIXME: rm 'sorted'

        self._int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self._class_to_int = {c: i for i, c in self._int_to_class.items()}

        self.y_train = self.transform_to_int_label(self.y_train)
        self.y_test = self.transform_to_int_label(self.y_test)

    def transform_to_int_label(self, y_split):
        labels_mod = []
        for label in y_split:
            label_mod = np.array([self._class_to_int[c] for c in label])
            if self.use_one_hot_labels is True:
                tmp = np.zeros((len(label_mod), len(self._int_to_class)))
                tmp[np.arange(len(label_mod)), label_mod] = 1
                label_mod = tmp
            labels_mod.append(label_mod)
        return np.array(labels_mod)

    def _load_internal(self, data_folder):
        train_x_y_pairs, val_x_y_pairs = load_all_data(
            data_folder, self.duration, self.val_frac, step=self.step)
        random.shuffle(train_x_y_pairs)

        x_train, y_train = zip(*train_x_y_pairs)
        x_test, y_test = zip(*val_x_y_pairs)

        return (x_train, x_test, y_train, y_test)

    def _load(self, data_folder, use_cached_if_available):
        """Run the pipeline to load the dataset.

        Returns the dataset with a train test split.
        """
        cached_filename = data_folder + '/cached'

        def check_cached_copy():
            return os.path.isfile(cached_filename)

        def load_cached():
            return joblib.load(cached_filename)

        def save_loaded(loaded):
            joblib.dump(loaded, cached_filename)

        if use_cached_if_available and check_cached_copy():
            print("Using cached copy of dataset...")
            loaded = load_cached()
        else:
            print("Loading dataset (not stored in cache)...")
            loaded = self._load_internal(data_folder)
            print("Saving to cache (this may take some time)...")
            try:
                save_loaded(loaded)
            except:
                print("Couldn't save cache...")

        (self.x_train, self.x_test, self.y_train, self.y_test) = loaded

    def train_generator(self):
        return self._batch_generate(self.x_train, self.y_train)

    def test_generator(self):
        return self._batch_generate(self.x_test, self.y_test)

    def _batch_generate(self, inputs, labels):
        batch_size = self.batch_size
        data_size = len(labels)
        for i in range(0, data_size - batch_size + 1, batch_size):
            batch_data = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield (batch_data, batch_labels)

    @property
    def output_dim(self):
        return len(self._int_to_class)


def load(args, params):
    dl = Loader(
        args.data_path,
        seed=params["seed"] if "seed" in params else 2016,
        use_cached_if_available=not args.refresh,
        normalizer=params["normalizer"] if "normalizer" in params else False,
        wavelet_fns=params["wavelet_fns"] if "wavelet_fns" in params else [],
        step=params["step"] if "step" else 200)
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument(
        "--refresh",
        help="whether to refresh cache",
        action="store_true")
    args = parser.parse_args()

    params = json.load(open(args.config_file, 'r'))

    dl = load(args, params)
    print("Length of training set {}".format(len(dl.x_train)))
    print("Length of validation set {}".format(len(dl.x_test)))
    print("Output dimension {}".format(dl.output_dim))
