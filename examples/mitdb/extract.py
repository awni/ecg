from __future__ import print_function

import cPickle as pickle
import glob
import numpy as np
import os
import subprocess

WFDB = "/deep/group/med/tools/wfdb-10.5.24/build/bin/"
DATA = "/deep/group/med/mitdb"

def extract_wave(idx):
    """
    Reads .dat file and returns in numpy array. Assumes 2 channels.  The
    returned array is n x 3 where n is the number of samples. The first column
    is the sample number and the second two are the first and second channel
    respectively.
    """
    rdsamp = os.path.join(WFDB, 'rdsamp')
    output = subprocess.check_output([rdsamp, '-r', idx], cwd=DATA)
    data = np.fromstring(output, dtype=np.int32, sep=' ')
    return data.reshape((-1, 3))

def extract_annotation(idx):
    """
    The annotation file column names are:
        Time, Sample #, Type, Sub, Chan, Num, Aux
    The Aux is optional, it could be left empty. Type is the beat type and Aux
    is the transition label.
    """
    rdann = os.path.join(WFDB, 'rdann')
    output = subprocess.check_output([rdann, '-r', idx, '-a', 'atr'], cwd=DATA)
    labels = (line.split() for line in output.strip().split("\n"))
    labels = [(l[0], int(l[1]), l[2], l[6] if len(l) == 7 else None)
                for l in labels]
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
    with open(os.path.join(DATA, "{}.pkl".format(idx)), 'w') as fid:
        pickle.dump(example[1], fid)

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA, "*.dat"))
    idxs = [os.path.basename(f).split(".")[0] for f in files]
    for idx in idxs:
        example = extract(idx)
        save(example, idx)
        print("Example {}".format(idx))
