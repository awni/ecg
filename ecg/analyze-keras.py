import argparse
import json
import os


def get_saved_models(path):
    models = []
    for (dirpath, dirnames, filenames) in os.walk(args.saved_path):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                loss = float(filename.split('-')[1].split('.hdf5')[0])
                models.append((loss, filename, dirpath))
    models.sort()
    return models


def get_best_model(path, get_structure=False):
    models = get_saved_models(path)
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
    print(get_best_model(args.saved_path, get_structure=True))
