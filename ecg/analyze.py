from __future__ import print_function
from builtins import zip
from builtins import open
import argparse
import csv
import json
import os
from tabulate import tabulate
from collections import defaultdict
from io import BytesIO


def get_params_table(path, max_models=5, metric="val_loss"):
    def process_params(parameters):
        for key in parameters:
            if isinstance(parameters[key], list):
                fq = defaultdict(int)
                for w in parameters[key]:
                    fq[w] += 1
                parameters[key] = dict(fq)
        if 'FOLDER_TO_SAVE' in parameters:
            del parameters["FOLDER_TO_SAVE"]
        return parameters

    output = BytesIO()
    first = True
    visited_dirs = {}
    for sortval, metric_table, _, dirpath in get_best_models(path, metric):
        if len(visited_dirs) == max_models:
            break
        if dirpath in visited_dirs:
            continue
        visited_dirs[dirpath] = True
        parameters = json.load(open(os.path.join(dirpath, 'params.json'), 'r'))
        parameters = process_params(parameters)
        parameters.update({"_val": sortval})
        parameters.update(metric_table)
        parameters.update({'dirpath': os.path.basename(dirpath)})
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


def get_best_models(path, metric='val_loss'):
    models = []
    for (dirpath, dirnames, filenames) in os.walk(args.saved_path):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                name_split = filename.split('.hdf5')[0].split('-')
                metric_table = {
                    'val_loss': float(name_split[0]),
                    'val_accuracy': float(name_split[1]),
                    'epoch': float(name_split[2]),
                    'loss': float(name_split[3]),
                    'accuracy': float(name_split[4])
                }
                sort_value = metric_table[metric]
                models.append((sort_value, metric_table, filename, dirpath))
    models.sort(reverse='accuracy' in metric)
    return models


def get_best_model(path, get_structure=False, metric='val_loss'):
    models = get_best_models(path, metric)
    best_model = models[0]
    dirpath = best_model[3]
    filename = best_model[2]
    best_model_path = os.path.join(dirpath, filename)
    if get_structure is True:
        structure = json.load(open(os.path.join(dirpath, 'params.json'), 'r'))
        return best_model_path, structure
    return best_model_path


def analyze(args):
    best_model_path = get_best_model(args.saved_path, metric=args.metric)
    print('Best model path', args.metric, ':', best_model_path)
    params_table = get_params_table(args.saved_path, metric=args.metric)
    print(params_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_path", help="path to saved files")
    parser.add_argument(
        "--metric",
        help="metric to use",
        default='val_loss',
        choices=['val_loss', 'loss', 'accuracy', 'val_accuracy'])
    args = parser.parse_args()
    analyze(args)
