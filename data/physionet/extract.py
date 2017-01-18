"""
Given a path to physionet ECG dataset this will extract all the raw ECG data
and annotations into easier to use file formats. ECG data gets serialized as
npy files and the labels are pickled.

Requires an installation of WFDB
    https://physionet.org/physiotools/wfdb.shtml
"""
from __future__ import print_function

import cPickle as pickle
import glob
import numpy as np
import os
import subprocess

WFDB = "/deep/group/med/tools/wfdb-10.5.24/build/bin/"

def extract_wave(idx, data_path):
    """
    Reads .dat file and returns in numpy array. Assumes 2 channels.  The
    returned array is n x 3 where n is the number of samples. The first column
    is the sample number and the second two are the first and second channel
    respectively.
    """
    rdsamp = os.path.join(WFDB, 'rdsamp')
    output = subprocess.check_output([rdsamp, '-r', idx], cwd=data_path)
    data = np.fromstring(output, dtype=np.int32, sep=' ')
    return data.reshape((-1, 3))

def extract_annotation(idx, suffix, data_path):
    """
    The annotation file column names are:
        Time, Sample #, Type, Sub, Chan, Num, Aux
    The Aux is optional, it could be left empty. Type is the beat type and Aux
    is the transition label.
    """
    rdann = os.path.join(WFDB, 'rdann')
    output = subprocess.check_output([rdann, '-r', idx, '-a', suffix], cwd=data_path)
    labels = (line.split() for line in output.strip().split("\n"))
    labels = [(l[0], int(l[1]), l[2], l[6] if len(l) == 7 else None)
                for l in labels]
    return labels

def extract(idx, suffix, data_path):
    """
    Extracts data and annotations from .dat and .atr files.
    Returns a numpy array for the data and a list of tuples for the labels.
    """
    data = extract_wave(idx, data_path)
    labels = extract_annotation(idx, suffix, data_path)
    return data, labels

def save(example, idx, data_path):
    """
    Saves data with numpy.save (load with numpy.load) and pickles labels. The
    files are saved in the same place as the raw data.
    """
    np.save(os.path.join(data_path, idx), example[0])
    with open(os.path.join(data_path, "{}.pkl".format(idx)), 'w') as fid:
        pickle.dump(example[1], fid)

def get_idxs(data_path):
    files = glob.glob(os.path.join(data_path, "*.dat"))
    return [os.path.basename(f).split(".")[0] for f in files]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to physionet data.")
    parser.add_argument("--suffix", default="atr", help="Annotation file suffix.")
    args = parser.parse_args()

    idxs = get_idxs(args.data_path)
    for idx in idxs:
        example = extract(idx, args.suffix, args.data_path)
        save(example, idx, args.data_path)
        print("Example {}".format(idx))
