import collections
import numpy as np
import os
import random
import joblib
from pprint import pprint
import argparse

from data.irhythm.extract_data import load_all_data


class Loader(object):
    """
    Loader class for feeding data to the network.
    """

    def __init__(self, data_path, batch_size, duration=30,
                 val_frac=0.1, seed=None, use_cached_if_available=True):
        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        self.batch_size = batch_size
        self.duration = duration
        self.val_frac = val_frac
        self._load(data_path, use_cached_if_available)
        self._postprocess()

    def _postprocess(self):
        label_counter = collections.Counter(l for labels in self.y_train
                                            for l in labels)
        pprint(label_counter)
        classes = sorted([c for c, _ in label_counter.most_common()])
        self._int_to_class = dict(zip(range(len(classes)), classes))
        self._class_to_int = {c: i for i, c in self._int_to_class.items()}

        self.y_train = self.transform_to_int_label(self.y_train)
        self.y_test = self.transform_to_int_label(self.y_test)

    def transform_to_int_label(self, y_split):
        return [[self._class_to_int[c] for c in label] for label in y_split]

    def _load_internal(self, data_folder):
        def normalize(example, mean, std):
            """
            Normalizes a given example by the training mean and std.
            :param: example: 1D numpy array
            :return: normalized example
            """
            return (example - mean) / std

        def compute_mean_std(data_pairs):
            """
            Estimates the mean and std over the training set.
            """
            all_dat = np.hstack(w for w, _ in data_pairs)
            mean = np.mean(all_dat)
            std = np.std(all_dat)
            return mean, std

        train_x_y_pairs, val_x_y_pairs = load_all_data(
            data_folder, self.duration, self.val_frac)
        random.shuffle(train_x_y_pairs)
        mean, std = compute_mean_std(train_x_y_pairs)
        train_x_y_pairs = [(
            normalize(ecg, mean, std), l) for ecg, l in train_x_y_pairs]
        val_x_y_pairs = [(
            normalize(ecg, mean, std), l) for ecg, l in val_x_y_pairs]

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
            save_loaded(loaded)

        (self.x_train, self.x_test, self.y_train, self.y_test) = loaded

    def train_generator(self):
        return self._batch_generate(self.x_train, self.y_train)

    def test_generator(self):
        return self._batch_generate(self.x_test, self.y_test)

    def _batch_generate(self, inputs, labels):
        """
        :param data: the raw dataset from e.g. `loader.train`
        :returns: Iterator to the minibatches. Each minibatch consists
                  of an (ecgs, labels) pair. The ecgs is a list of 1D
                  numpy arrays, the labels is a list of integer labels
                  for each ecg.
        """
        batch_size = self.batch_size
        data_size = len(labels)
        for i in range(0, data_size - batch_size + 1, batch_size):
            batch_data = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield (batch_data, batch_labels)

    @property
    def output_dim(self):
        """ Returns number of output classes. """
        return len(self._int_to_class)


if __name__ == "__main__":
    random.seed(2016)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("--refresh", help="whether to refresh cache")
    args = parser.parse_args()
    batch_size = 32
    ldr = Loader(
        args.data_path,
        batch_size,
        use_cached_if_available= not args.refresh)
    print("Length of training set {}".format(len(ldr.x_train)))
    print("Length of validation set {}".format(len(ldr.x_test)))
    print("Output dimension {}".format(ldr.output_dim))

    # Run a few sanity checks.
    count = 0
    for ecgs, labels in ldr.train_generator():
        count += 1
        assert len(ecgs) == len(labels) == batch_size, \
            "Invalid number of examples."
        assert len(ecgs[0].shape) == 1, "ECG array should be 1D"
    assert count == len(ldr.x_train) // batch_size, "Wrong number of batches."
