from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import open
import argparse
import numpy as np

import load
import util


def predict(args, params):
    dl = load.load(args, params)
    from keras.models import load_model
    model = load_model(args.model_path)
    split = args.split
    x_val = dl.x_train if split == 'train' else dl.x_test
    print("Predicting on:", split)
    predictions = model.predict(x_val, verbose=1)
    with open(util.get_prediction_path_for_model(
              args.model_path, split), 'wb') as outfile:
        np.save(outfile, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("model_path", help="path to model")
    parser.add_argument("split", help="train/val", choices=['train', 'test'])
    args = parser.parse_args()
    params = util.get_model_params(args.model_path)
    predict(args, params)
