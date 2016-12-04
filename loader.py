from __future__ import division
from __future__ import print_function

import cPickle as pickle
import collections
import glob
import numpy as np
import os
import random

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

    # TODO, awni, don't hard code this here, didn't want to re-preprocess the data.
    FILTER_LABELS = ['SVT', 'BIGEMINY', 'TRIGEMINY', 'NOISE']

    def __init__(self, data_path, batch_size):
        """
        :param data_path: path to the training and validation files
        :param batch_size: size of the minibatches to train on
        :param rng_seed: seed the rng for shuffling data
        """
        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        self.batch_size = batch_size

        self._train = _read_dataset(data_path, "train")
        self._val = _read_dataset(data_path, "val")

        def filter_fn(dset):
            return filter(lambda x : x[1] not in Loader.FILTER_LABELS, dset)

        self._train = filter_fn(self._train)
        self._val = filter_fn(self._val)

        random.shuffle(self._train)

        self.compute_mean_std()
        self._train = [(self.normalize(ecg), l) for ecg, l in self._train]
        self._val = [(self.normalize(ecg), l) for ecg, l in self._val]

        # Can use this to look at the distribution of classes
        # for each rhythm.
        label_counter = collections.Counter(l for _, l in self._train)

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
        labels = [self._class_to_int[c] for c in labels]
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

def _read_dataset(data_path, dataset):
    """
    Reads an irhythm dataset into a list of examples.
    :param data_path: specifies the top level path of the data.
    :param dataset: specifies the train / val / test data.
    :return: A list of examples. Each example is an ecg, label pair.
             The ecg is a numpy array and the label is a string.
    """
    ecg_pattern = os.path.join(data_path, dataset, "*.npy")
    ecg_files = glob.glob(ecg_pattern)
    examples = []
    for ecg_file in ecg_files:
        inputs = np.load(ecg_file)
        label_file = os.path.splitext(ecg_file)[0] + ".pkl"
        with open(label_file, 'r') as fid:
            label = pickle.load(fid)
        examples.append((inputs, label))
    return examples

if __name__ == "__main__":
    data_path = "/deep/group/med/irhythm/ecg/extracted_clean_all"
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


