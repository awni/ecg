import os
import json


def get_confusion_figure_path(model_path):
    return model_path + '.png'


def get_prediction_path_for_model(model_path, split):
    return model_path + '-pred-' + split + '.pkl'


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
