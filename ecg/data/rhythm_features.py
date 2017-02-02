from __future__ import print_function

import json
import os
import numpy as np

YES = "yes"
NO = "no"
DONT_CARE = "x"
DIR_NAME = os.path.dirname(os.path.abspath(__file__))
DOMAIN_FILE = os.path.join(DIR_NAME, "ecg_domain.json")

def binarize(rhythm, features):
    feat_vec = np.empty(len(features))
    mask = np.ones(len(features))
    for e, feat in enumerate(features):
        if rhythm[feat] == YES:
            feat_vec[e] = 1
        elif rhythm[feat] == NO:
            feat_vec[e] = 0
        elif rhythm[feat] == DONT_CARE:
            feat_vec[e] = 0
            mask[e] = 0
        else:
            raise ValueError("Invalid feature value")
    return feat_vec, mask

def load_domain():
    with open(DOMAIN_FILE, 'r') as fid:
        domain_theory = json.load(fid)
    features = domain_theory['features']
    rhythms = domain_theory['rhythms']
    for rhythm in rhythms:
        rhythm = rhythms[rhythm]
        feats, mask = binarize(rhythm, features)
        rhythm['binary_feats'] = feats
        rhythm['mask'] = mask
    return domain_theory

def get_features(rhythm, domain_theory):
    rhythm = domain_theory['rhythms'][rhythm]
    return rhythm['binary_feats'], rhythm['mask']

def num_features(domain_theory):
    return len(domain_theory['features'])

if __name__ == "__main__":
    domain = load_domain(DOMAIN_FILE)
