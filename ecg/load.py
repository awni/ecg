from __future__ import division
from __future__ import print_function
from builtins import dict
from builtins import zip
from builtins import range
import argparse
import collections
import numpy as np
import os
import json
from joblib import Memory
import featurize
from data.irhythm.extract_data import load_all_data
from data import rhythm_features

memory = Memory(cachedir='./data_cache', verbose=1)

class Loader(object):
    def __init__(
            self,
            data_path,
            batch_size=32,
            duration=30,
            step=200,
            val_frac=0.1,
            seed=2016,
            use_one_hot_labels=True,
            use_cached_if_available=True,
            save_cache_if_possible=True,
            normalizer=False,
            ignore_classes=[],
            wavelet_fns=[],
            wavelet_type='discrete',
            wavelet_level=1,
            use_bandpass_filter=False,
            toy=False,
            **kwargs):

        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        np.random.seed(seed)
        self.batch_size = batch_size
        self.duration = duration
        self.val_frac = val_frac
        self.wavelet_fns = wavelet_fns
        self.normalizer = normalizer
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.use_one_hot_labels = use_one_hot_labels
        self.step = step
        self.ignore_classes = ignore_classes
        self.use_cached_if_available = use_cached_if_available
        self.save_cache_if_possible = save_cache_if_possible
        self.use_bandpass_filter = use_bandpass_filter
        self.toy = toy

        (self.x_train, self.x_test, self.y_train, self.y_test) = \
            self._load_internal(data_path)
        self._postprocess()

    def _postprocess(self):
        self.x_train = np.array(self.x_train)[..., None]
        self.x_test = np.array(self.x_test)[..., None]
        rhythm_feats = self._rhythm_feature_labels(self.y_train)
        self.rfeats_train, self.mask_train = rhythm_feats
        rhythm_feats = self._rhythm_feature_labels(self.y_test)
        self.rfeats_val, self.mask_val = rhythm_feats

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        if self.use_bandpass_filter is True:
            bp_filter = featurize.BandPassFilter()
            self.x_train = bp_filter.transform(self.x_train)
            self.x_test = bp_filter.transform(self.x_test)

        if len(self.wavelet_fns) != 0:
            if (self.wavelet_type == 'discrete'):
                wavelet_transformer = \
                    featurize.DiscreteWaveletTransformer(
                        self.wavelet_fns, self.wavelet_level)
            elif (self.wavelet_type == 'continuous'):
                wavelet_transformer = \
                    featurize.ContinuousWaveletTransformer(self.wavelet_fns)
            else:
                raise ValueError("Wavelet type not defined.")
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

        if self.ignore_classes is not False:
            for ignore_class in self.ignore_classes:
                print("Ignoring class: " + ignore_class)
                for split in ['_train', '_test']:
                    indices = np.where(np.sum(getattr(
                        self, 'y' + split) == ignore_class, axis=1) == 0)[0]
                    for prop in ['x', 'y']:
                        setattr(self, prop + split, getattr(
                            self, prop + split)[indices])

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

    def _rhythm_feature_labels(self, y_split):
        domain_theory = rhythm_features.load_domain()
        num_feats = rhythm_features.num_features(domain_theory)
        rfeats = np.empty((len(y_split), len(y_split[0]), num_feats))
        mask = np.empty((len(y_split), len(y_split[0]), num_feats))
        for i, y in enumerate(y_split):
            for j, l in enumerate(y):
                f, m = rhythm_features.get_features(l,
                                        domain_theory)
                rfeats[i, j, :] = f
                mask[i, j, :] = m
        return rfeats, mask

    def _load_internal(self, data_folder):
        train_x_y_pairs, val_x_y_pairs = load_all_data(
            data_folder,
            self.duration,
            self.val_frac,
            step=self.step,
            toy=self.toy)

        x_train, y_train = zip(*train_x_y_pairs)
        x_test, y_test = zip(*val_x_y_pairs)

        return (x_train, x_test, y_train, y_test)

    def _load(self, data_folder):
        """Run the pipeline to load the dataset.

        Returns the dataset with a train test split.
        """
        cached_filename = data_folder + '/cached-' + str(self.step)

        def check_cached_copy():
            return os.path.isfile(cached_filename)

        def load_cached():
            return joblib.load(cached_filename)

        def save_loaded(loaded):
            joblib.dump(loaded, cached_filename)

        if self.use_cached_if_available is True and check_cached_copy():
            print("Using cached copy of dataset...")
            loaded = load_cached()
        else:
            print("Loading dataset...")
            loaded = self._load_internal(data_folder)
            if self.save_cache_if_possible is True:
                print("Saving to cache (this may take some time)...")
                try:
                    save_loaded(loaded)
                except:
                    print("Couldn't save cache...")

        (self.x_train, self.x_test, self.y_train, self.y_test) = loaded

    @property
    def output_dim(self):
        return len(self._int_to_class)

    @property
    def class_to_int(self):
        return self._class_to_int


@memory.cache
def load_inner(data_path, params):
    dl = Loader(data_path, **params)
    return dl


def load(args, params):
    dl = load_inner(args.data_path, params)
    print("Length of training set {}".format(len(dl.x_train)))
    print("Length of validation set {}".format(len(dl.x_test)))
    print("Output dimension {}".format(dl.output_dim))
    return dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    load(args, params)
