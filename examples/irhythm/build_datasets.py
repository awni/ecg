from __future__ import division
from __future__ import print_function

import argparse
import fnmatch
import glob
import json
import numpy as np
import os
import random
import tqdm

STEP = 256
RELABEL = {"NSR": "SINUS", "SUDDEN_BRADY": "AVB",
           "AVB_TYPE2": "AVB", "AFIB": "AF", "AFL": "AF"}

def get_all_records(path, blacklist=set()):
    records = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.ecg'):
            if patient_id(filename) not in blacklist:
                records.append(os.path.abspath(
                    os.path.join(root, filename)))
    return records

def patient_id(record):
    return os.path.basename(record).split("_")[0]

def round_to_step(n, step):
    diff = (n - 1) % step 
    if diff < (step / 2):
        return n - diff
    else:
        return n + (step - diff)

def load_episodes(record, epi_ext):

    base = os.path.splitext(record)[0]
    ep_json = base + epi_ext
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        episodes = json.load(fid)['episodes']
    episodes = sorted(episodes, key=lambda x: x['onset'])

    for episode in episodes:
        episode['onset_round'] = round_to_step(episode['onset'], STEP)
        rn = episode['rhythm_name']
        episode['rhythm_name'] = RELABEL.get(rn, rn)

    for e, episode in enumerate(episodes):
        if e == len(episodes) - 1:
            episode['offset_round'] = episode['offset']
        else:
            episode['offset_round'] = episodes[e+1]['onset_round'] - 1
    return episodes

def make_labels(episodes):
    labels = []
    for episode in episodes:
        rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
        rhythm_labels = int(rhythm_len / STEP)
        rhythm = [episode['rhythm_name']] * rhythm_labels
        labels.extend(rhythm)
    trunc_samp = int(episodes[-1]['offset'] / STEP)
    labels = labels[:trunc_samp]
    return labels

def build_blacklist(blacklist_paths):
    print('Building blacklist...')
    blacklist = set()
    for blacklist_path in blacklist_paths:
        print(blacklist_path)
        for record in get_all_records(blacklist_path):
            blacklist.add(patient_id(record))
    return blacklist

def construct_dataset(records, epi_ext='.episodes.json'):
    data = []
    for record in tqdm.tqdm(records):
        labels = make_labels(load_episodes(record, epi_ext))
        assert len(labels) != 0, "Zero labels?"
        data.append((record, labels))
    return data

def stratify(records, dev_frac):
    pids = list(set(patient_id(record) for record in records))
    random.shuffle(pids)
    cut = int(len(pids) * dev_frac)
    dev_pids = set(pids[:cut])
    train = [r for r in records if patient_id(r) not in dev_pids] 
    dev = [r for r in records if patient_id(r) in dev_pids] 
    return train, dev 

def load_train(data_path, dev_frac, blacklist_paths):
    blacklist = build_blacklist(blacklist_paths)
    records = get_all_records(data_path, blacklist)
    train, dev = stratify(records, dev_frac)
    print("Constructing train...")
    train = construct_dataset(train)
    print("Constructing dev...")
    dev = construct_dataset(dev)
    return train, dev

def load_rev_id(record, epi_ext):

    base = os.path.splitext(record)[0]
    ep_json = base + epi_ext
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        return json.load(fid)['reviewer_id']

def load_test(data_path, epi_ext):
    records = get_all_records(data_path)
    print("Constructing test...")
    test = construct_dataset(records, epi_ext)
    # Get the reviewer id
    reviewers = [load_rev_id(r, epi_ext) for r in records]
    test = [(e, l, r)
            for (e, l), r in zip(test, reviewers)]
    return test

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            if len(d) == 3:
                datum['reviewer'] = d[2]
            json.dump(datum, fid)
            fid.write('\n')


if __name__ == "__main__":
    data_dir = "/deep/group/med/irhythm/ecg/clean_30sec_recs/"
    blacklist_paths = [
            os.path.join(data_dir, "label_review/CARDIOL_MAY_2017/"),
            os.path.join(data_dir, "batches/kids_blacklist"),
            os.path.join(data_dir, "batches/vf_blacklist")]
    data_path = os.path.join(data_dir, "batches")
    dev_frac = 0.1
    train, dev = load_train(data_path, dev_frac, blacklist_paths)
    make_json("train.json", train)
    make_json("dev.json", dev)
    test_dir = os.path.join(data_dir, "label_review/CARDIOL_UNIQ_P/")
    test = load_test(test_dir, '_grp*.episodes.json')
    make_json("test.json", test)
    for i in range(6):
        test = load_test(test_dir, "_rev{}.episodes.json".format(i))
        make_json("test_rev{}.json".format(i), test)
