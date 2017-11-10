
import glob
import os
import random
import json
import numpy as np

path = "/deep/group/med/irhythm/ecg/clean_30sec_recs/label_review/CE_APR_2017"
ecg_files = glob.glob(os.path.join(path, "*.ecg"))

keys = [os.path.basename(e).split(".")[0] for e in ecg_files]

random.shuffle(keys)

data = []
labels = ['AFIB', 'NSR', 'BIGEMINY', 'TRIGEMINY', 'SVT',
          'VT', 'SUDDEN_BRADY', 'AVB_TYPE2', 'WENCKEBACH', 'AFL',
          'IVR', 'JUNCTIONAL', 'NOISE', 'EAR']
label_ids = dict(enumerate(labels))
labels = dict((v, k) for k,v in label_ids.items())
for k in keys[:20]:
    ecg_file = os.path.join(path, k + ".ecg")
    label_file = os.path.join(path, k + "_rev0.episodes.json")
    with open(ecg_file, 'r') as fid:
        ecg = np.fromfile(fid, dtype=np.int16)
    with open(label_file, 'r') as fid:
        episodes = json.load(fid)['episodes']
    ecg = (ecg - np.mean(ecg)) / np.std(ecg)
    ecg = ecg.tolist()
    eps = []
    for episode in episodes:
        rhythm = episode['rhythm_name']
        duration = episode['offset'] - episode['onset'] + 1
        eps.extend([labels[rhythm]] * duration)
    assert len(eps) == len(ecg), "Bad length"
    data.append({'x' : ecg, 'y' : eps})

json_dat = {'data' : data, 'labels' : label_ids}
with open("data.json", 'w') as fid:
    json.dump(json_dat, fid)
