import json
import numpy as np
import os
import random
import scipy.io as sio
import scipy.signal as ssi
import tqdm

STEP = 256

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def resample_and_save(data_path, record):
    ecg_file = os.path.join(data_path, record + ".mat")
    ecg = load_ecg_mat(ecg_file)
    ecg = ssi.resample(ecg, int((200 / 300.) * ecg.size))
    ecg = ecg.astype(np.float32)
    new_ecg_file = os.path.join(data_path, record + ".npy")
    np.save(new_ecg_file, ecg) 
    return ecg, new_ecg_file

def load_all(data_path):
    label_file = os.path.join(data_path, "REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg, ecg_file = resample_and_save(data_path, record)
        num_labels = ecg.shape[0] / STEP
        dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def split(dataset, dev_frac, test_frac):
    dev_cut = int(dev_frac * len(dataset))
    test_cut = dev_cut + int(test_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    test = dataset[dev_cut:test_cut]
    train = dataset[test_cut:]
    return train, dev, test

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    dev_frac = 0.1
    test_frac = 0.1
    data_path = "/deep/group/med/alivecor/training2017/"
    dataset = load_all(data_path)
    train, dev, test = split(dataset, dev_frac, test_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)
    make_json("test.json", test)

