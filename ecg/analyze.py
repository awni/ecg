from __future__ import print_function
from builtins import zip
from builtins import open
from builtins import str
import argparse
import csv
import json
import os
from tabulate import tabulate
from io import BytesIO

DEFAULT_VERSION = 2


def get_params_table(path, max_models=5, version=DEFAULT_VERSION,
                     metric="val_loss"):

    def process_params(parameters):
        for key in parameters:
            if isinstance(parameters[key], list):
                parameters[key] = ','.join(str(x) for x in parameters[key])
        if 'FOLDER_TO_SAVE' in parameters:
            del parameters["FOLDER_TO_SAVE"]
        return parameters

    output = BytesIO()
    first = True
    visited_dirs = {}
    for loss, _, dirpath in get_best_models(path, version, metric):
        if len(visited_dirs) == max_models:
            break
        if dirpath in visited_dirs:
            continue
        visited_dirs[dirpath] = True
        parameters = json.load(open(os.path.join(dirpath, 'params.json'), 'r'))
        parameters = process_params(parameters)
        parameters.update({"_loss": loss})
        if first is True:
            fieldnames = sorted(parameters.keys())
            writer = csv.DictWriter(
                        output,
                        fieldnames=fieldnames,
                        extrasaction='ignore')
            writer.writeheader()
            first = False
        writer.writerow(parameters)
    output.seek(0)
    return tabulate(list(zip(*csv.reader(output))))


def get_best_models(path, version=DEFAULT_VERSION, metric='val_loss'):
    models = []
    for (dirpath, dirnames, filenames) in os.walk(args.saved_path):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                name_split = filename.split('.hdf5')[0].split('-')
                if version == 1:
                    assert(metric == 'val_loss')
                    loss = float(name_split[1])
                elif version == 2:
                    if metric == 'val_loss':
                        loss = float(name_split[0])
                    elif metric == 'loss':  # train loss
                        loss = float(name_split[-2])
                    else:
                        raise ValueError('Metric not defined')
                else:
                    raise ValueError('Version not defined')
                models.append((loss, filename, dirpath))
    models.sort()
    return models


def get_best_model(path, get_structure=False, version=DEFAULT_VERSION,
                   metric='val_loss'):
    models = get_best_models(path, version, metric)
    best_model = models[0]
    dirpath = best_model[2]
    filename = best_model[1]
    best_model_path = os.path.join(dirpath, filename)
    if get_structure is True:
        structure = json.load(open(os.path.join(dirpath, 'params.json'), 'r'))
        return best_model_path, structure
    return best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_path", help="path to saved files")
    parser.add_argument(
        "--version",
        help="version of saved files",
        default=DEFAULT_VERSION,
        type=int)
    parser.add_argument(
        "--metric",
        help="metric to use",
        default='val_loss',
        choices=['val_loss', 'loss'])
    args = parser.parse_args()
    print('Best model path', args.metric, ':', get_best_model(
            args.saved_path, version=args.version, metric=args.metric))
    print(get_params_table(
        args.saved_path, version=args.version, metric=args.metric))
