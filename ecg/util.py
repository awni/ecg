import os
import cPickle as pickle
import json

def get_model_params(model_path):
    params = json.load(open(
        os.path.dirname(model_path) + '/params.json', 'r'))
    return params

def get_object_from_dict(**params):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    args = AttrDict(**params)
    return args

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'r') as fid:
        preproc = pickle.load(fid)
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'w') as fid:
        pickle.dump(preproc, fid)
