from __future__ import print_function

import argparse
import numpy as np
import json
import keras
import os

import load
import util

def get_model_probs(model_path, x):
    return probs

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    print(probs.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    predict(args.data_json, args.model_path)
