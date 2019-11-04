from __future__ import print_function

import json
import random
import pickle
import glob
import numpy as np
import os
import subprocess

WFDB = "/deep/group/med/tools/wfdb-10.5.24/build/bin/"
DATA = "/Users/mac/Desktop/wellysis/ecg/examples/mitdb/data/"

def extract_wave(idx):
    """
    Reads .dat file and returns in numpy array. Assumes 2 channels.  The
    returned array is n x 3 where n is the number of samples. The first column
    is the sample number and the second two are the first and second channel
    respectively.
    """
    # rdsamp = os.path.join(WFDB, 'rdsamp')
    output = subprocess.check_output(['rdsamp', '-r', idx], cwd=DATA)
    data = np.fromstring(output, dtype=np.int32, sep=' ')
    data = data.reshape((-1, 3))
    # only extract lead 2
    return data[:, 1]

def extract_annotation(idx):
    """
    The annotation file column names are:
        Time, Sample #, Type, Sub, Chan, Num, Aux
    The Aux is optional, it could be left empty. Type is the beat type and Aux
    is the transition label.
    """
    # rdann = os.path.join(WFDB, 'rdann')
    output = subprocess.check_output(['rdann', '-r', idx, '-a', 'atr'], cwd=DATA)
    labels = (line.split() for line in output.strip().split(b"\n"))
    # labels = [(l[0].decode('ascii'), int(l[1]), l[2].decode('ascii'), l[6].decode('ascii') if len(l) == 7 else None)
    #             for l in labels]
    labels = [l[2].decode('ascii') for l in labels]
    return labels

def extract(idx):
    """
    Extracts data and annotations from .dat and .atr files.
    Returns a numpy array for the data and a list of tuples for the labels.
    """
    data = extract_wave(idx)
    labels = extract_annotation(idx)
    return data, labels

def save(example, idx):
    """
    Saves data with numpy.save (load with numpy.load) and pickles labels. The
    files are saved in the same place as the raw data.
    """
    np.save(os.path.join(DATA, idx), example[0])
    with open(os.path.join(DATA, "{}.pkl".format(idx)), 'wb') as fid:
        # if idx == '100':
        #     print(example[1])
        pickle.dump(example[1], fid)
    return os.path.join(DATA, idx)+'.npy', os.path.join(DATA, idx)+'.pkl'

def load_all(data_path):
    files = glob.glob(os.path.join(DATA, "*.dat"))
    idxs = [os.path.basename(f).split(".")[0] for f in files]

    dataset = []
    for idx in idxs:
        example = extract(idx)
        npy, pkl = save(example, idx)
        dataset.append((npy, pkl))
        print("Example {}".format(idx))
    return dataset

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg': d[0],
                     'labels': d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    random.seed(2019)

    dev_frac = 0.1
    dataset= load_all(DATA)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)
