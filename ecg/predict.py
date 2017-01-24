from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import open
import argparse
import numpy as np
import json
import os
from keras.models import load_model

import load

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("model_path", help="path to model")
    parser.add_argument(
        "--refresh",
        help="whether to refresh cache",
        action="store_true")
    args = parser.parse_args()

    params = json.load(open(
        os.path.dirname(args.model_path) + '/params.json', 'r'))

    dl = load.load(args, params)

    model = load_model(args.model_path)
    for x, name in [(dl.x_train, 'train'), (dl.x_test, 'test')]:
        print("Predicting on:", name)
        predictions = model.predict(x, verbose=1)
        with open(args.model_path + '-pred-' + name + '.pkl', 'wb') as outfile:
            np.save(outfile, predictions)
