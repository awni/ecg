from __future__ import division
from __future__ import print_function

import cPickle as pickle
import collections
import glob
import numpy as np
import os
import random
from data.irhythm.extract_data import load_all_data

class Loader:
    """
    Loader class for feeding data to the network. This class loads the training
    and validation data sets. Once the datasets are loaded, they can be batched
    and fed to the network. Example usage:

        ```
        data_path = <path_to_data>
        batch_size = 32
        ldr = Loader(data_path, batch_size)
        for batch in ldr.batches(ldr.train):
            run_sgd_on(batch)
        ```

    At the moment we expect the location where the data is stored to have
    train/ and val/ directories.  This class is also responsible for
    normalizing the inputs.
    """

    FILTER_LABELS = set(['SVT', 'BIGEMINY', 'TRIGEMINY', 'NOISE'])

    def __init__(self, data_path, batch_size, duration=30,
                 val_frac=0.1, seed=None):
        """
        :param data_path: path to the training and validation files
        :param batch_size: size of the minibatches to train on
        :param duration: length in seconds of an ecg example
        :param val_frac: fraction of the dataset to use for validation
                         (held out by record)
        :param seed: seed the rng for shuffling data
        """
        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        self.batch_size = batch_size

        self._train, self._val = load_all_data(data_path,
                                   duration, val_frac)
        def filter_fn(example):
           return all(l not in Loader.FILTER_LABELS
                        for l in example[1])

        self._train = filter(filter_fn, self._train)
        self._val = filter(filter_fn, self._val)

        random.shuffle(self._train)
        self.compute_mean_std()
        self._train = [(self.normalize(ecg), l) for ecg, l in self._train]
        self._val = [(self.normalize(ecg), l) for ecg, l in self._val]

        # Can use this to look at the distribution of classes
        # for each rhythm.
        label_counter = collections.Counter(l for _, labels in self._train
                                                 for l in labels)

        classes = [c for c, _ in label_counter.most_common()]

        # TODO, awni, these should be serialized with the loader.
        self._int_to_class = dict(zip(xrange(len(classes)), classes))
        self._class_to_int = {c : i for i, c in self._int_to_class.iteritems()}

    def batches(self, data):
        """
        :param data: the raw dataset from e.g. `loader.train`
        :returns: Iterator to the minibatches. Each minibatch consists
                  of an (ecgs, labels) pair. The ecgs is a list of 1D
                  numpy arrays, the labels is a list of integer labels
                  for each ecg.
        """
        inputs, labels = zip(*data)
        labels = [[self._class_to_int[c] for c in label]
                    for label in labels]
        batch_size = self.batch_size
        data_size = len(labels)
        for i in range(0, data_size - batch_size + 1, batch_size):
            batch_data = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield (batch_data, batch_labels)

    def normalize(self, example):
        """
        Normalizes a given example by the training mean and std.
        :param: example: 1D numpy array
        :return: normalized example
        """
        return (example - self.mean) / self.std

    def compute_mean_std(self):
        """
        Estimates the mean and std over the training set.
        """
        all_dat = np.hstack(w for w, _ in self._train)
        self.mean = np.mean(all_dat)
        self.std = np.std(all_dat)

    @property
    def output_dim(self):
        """ Returns number of output classes. """
        return len(self._int_to_class)

    @property
    def train(self):
        """ Returns the raw training set. """
        return self._train

    @property
    def val(self):
        """ Returns the raw validation set. """
        return self._val


if __name__ == "__main__":
    random.seed(2016)
    data_path = "/deep/group/med/irhythm/ecg/clean_5min_recs"
    batch_size = 32
    ldr = Loader(data_path, batch_size)
    print("Length of training set {}".format(len(ldr.train)))
    print("Length of validation set {}".format(len(ldr.val)))
    print("Output dimension {}".format(ldr.output_dim))

    # Run a few sanity checks.
    count = 0
    for ecgs, labels in ldr.batches(ldr.train):
        count += 1
        assert len(ecgs) == len(labels) == batch_size, \
                "Invalid number of examples."
        assert len(ecgs[0].shape) == 1, "ECG array should be 1D"
    assert count == len(ldr.train) // batch_size, \
            "Wrong number of batches."


