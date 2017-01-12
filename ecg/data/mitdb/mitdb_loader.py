"""
Deprecated file. We can back-port this later if we have reason to use the mitdb
dataset again.
"""
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import glob
import numpy as np
import os
import random

class Loader:

    max_len = 500

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.data_path = data_path
        wave_files = glob.glob(os.path.join(data_path, "*.npy"))
        patients = [os.path.basename(p).split(".")[0] for p in wave_files]
        patient_waves = [np.load(f) for f in wave_files]
        patient_annotations = []
        for p in patients:
            ann_file = os.path.join(data_path, "{}.pkl".format(p))
            with open(ann_file, 'r') as fid:
                patient_annotations.append(pickle.load(fid))
        patients = zip(patient_waves, patient_annotations)

        self._train = segment_all(patients[:40])
        # We *may* want to consider sorting by input length for efficiency
        # reasons. Though this could bias minibatches since examples with a
        # similar length are more likely to come from the same person since
        # they have a consistent heart rhythm
        random.shuffle(self._train)

        self._valid = segment_all(patients[40:44])
        self._test = segment_all(patients[44:])
        classes = set(c for _, c in self._train)
        self._vocab_size = len(classes)
        # TODO, these need to be serialized no guarantee on order
        self._int_to_label = dict(enumerate(classes))
        self._label_to_int = {l : i for i, l in self._int_to_label.iteritems()}

        self._train = self.preprocess(self._train)
        self._train = filter(lambda x : x[0].shape[0] < Loader.max_len,
                             self._train)
        self._valid = self.preprocess(self._valid)
        self.compute_mean_std()
        self.normalize(self._train)
        self.normalize(self._valid)
        # TODO, awni, some classes in test but not in train
        #self._test = self.preprocess(self._test)

    def batches(self, data):
        inputs, labels = zip(*data)
        batch_size = self.batch_size
        data_size = len(labels)
        for i in range(0, data_size - batch_size + 1, batch_size):
            batch_data = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield (batch_data, batch_labels)

    def normalize(self, dataset):
        for example in dataset:
            example[0][:] = (example[0] - self.mean) / self.std

    def compute_mean_std(self):
        """
        Estimates the mean and std over the training set.
        """
        all_dat = np.hstack(w for w, _ in self._train)
        self.mean = np.mean(all_dat)
        self.std = np.std(all_dat)

    @property
    def vocab_size(self):
        return self._vocab_size

    def preprocess(self, examples):
        """
        Select the first channel and convert labels to integer ids.
        """
        return [(w[:,1], self._label_to_int[l]) for w, l in examples]

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

def segment_all(patients):
    return [example for p in patients
                    for example in segment(*p)]

def segment(wave, labels):
    # TODO, awni, some of the labels aren't actually beats.
    for e, label in enumerate(labels):
        # Skip first and last
        if e == 0 or e == (len(labels) - 1): continue

        _, idx, beat, _ = label
        prev_idx = labels[e - 1][1]
        next_idx =  labels[e + 1][1]
        start = int((idx + prev_idx) / 2)
        end = int((idx + next_idx) / 2)
        yield (wave[start:end], beat)

if __name__ == "__main__":
    batch_size = 32
    ldr = Loader("/deep/group/med/mitdb", batch_size)
    print("Training set size: {}".format(len(ldr.train)))
    print("Validation set size: {}".format(len(ldr.valid)))
    print("Test set size: {}".format(len(ldr.test)))
    lengths = [w.shape[0] for w, l in ldr.train]
    print("Input lengths: Mean {:.1f}, Std {:.1f}, Min {}, Max {}".format(
        np.mean(lengths), np.std(lengths), min(lengths), max(lengths)))
    for e, (inputs, labels) in enumerate(ldr.batches(ldr.train)):
        assert len(inputs) == batch_size, "Bad inputs size."
        assert len(labels) == batch_size, "Bad labels size."
    assert (e + 1) == int(len(ldr.train) / batch_size), \
            "Bad number of batches."
