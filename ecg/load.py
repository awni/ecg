from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import tqdm

STEP = 256

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = set(l for label in labels for l in label)
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        x = np.array(x)
        x = (x - self.mean) / self.std
        x = x[:,:, None]
        y = np.array([[self.class_to_int[c] for c in s] for s in y])
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return x, y

def compute_mean_std(x):
    x = np.array(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    with open(record, 'r') as fid:
        ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

if __name__ == "__main__":
    data_json = "saved/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    x, y = preproc.process(*train)
    print(x.shape, y.shape)
    print(x.dtype)
