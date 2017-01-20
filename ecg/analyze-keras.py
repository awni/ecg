from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import csv
import json
import os
from tabulate import tabulate
from io import BytesIO


def get_params_table(path, max_models=5):
    def process_params(parameters):
        parameters['conv_subsample_lengths'] = ','.join(str(x) for x in parameters['conv_subsample_lengths'])
        if 'FOLDER_TO_SAVE' in parameters:
            del parameters["FOLDER_TO_SAVE"]
        return parameters

    output = BytesIO()
    first = True
    visited_dirs = {}
    for loss, _, dirpath in get_best_models(path):
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


def get_best_models(path):
    models = []
    for (dirpath, dirnames, filenames) in os.walk(args.saved_path):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                loss = float(filename.split('-')[1].split('.hdf5')[0])
                models.append((loss, filename, dirpath))
    models.sort()
    return models


def get_best_model(path, get_structure=False):
    models = get_best_models(path)
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
    args = parser.parse_args()
    print('Best model path: ', get_best_model(args.saved_path))
    print(get_params_table(args.saved_path))
