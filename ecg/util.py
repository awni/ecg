import os
import json
import time

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
