import json
import argparse
import numpy as np
import collections

import load


def stats(args, params):
    dl = load.load(args, params)
    y_train = dl.y_train
    y_val_flat = np.array(dl.classes)[np.argmax(y_train, axis=-1).flatten()]
    counter = collections.Counter(y_val_flat)
    print(counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data files")
    parser.add_argument("config_file", help="path to confile file")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    stats(args, params)
