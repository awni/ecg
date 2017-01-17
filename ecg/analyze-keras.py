import argparse
import csv
import json
import os
from tabulate import tabulate
from io import StringIO


def get_params_table(path, max_models=5):
    def process_params(parameters):
        parameters['subsample_lengths'] = ','.join(str(x) for x in parameters['subsample_lengths'])
        if 'FOLDER_TO_SAVE' in parameters:
            del parameters["FOLDER_TO_SAVE"]
        return parameters

    output = StringIO()
    first = True
    visited_dirs = {}
    for loss, _, dirpath in get_best_models(path):
        if visited_dirs == max_models:
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
    print(get_params_table(args.saved_path))
