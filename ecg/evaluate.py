from __future__ import print_function
from builtins import range
from builtins import open
from builtins import str
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm
import json
import os

import load
import decode
import util


def evaluate(args, params):
    dl = load.load(args, params)
    split = args.split
    x_val = dl.x_train if split == 'train' else dl.x_test
    y_val = dl.y_train if split == 'train' else dl.y_test
    print("Size: " + str(len(x_val)) + " examples.")

    predictions = np.load(open(util.get_prediction_path_for_model(
                               args.model_path, split), 'rb'))

    if args.decode is True:
        language_model = decode.LM(dl.y_train, dl.output_dim, order=2)
        predictions = np.array([decode.beam_search(prediction, language_model)
                                for prediction in tqdm(predictions)])
    else:
        predictions = np.argmax(predictions, axis=-1)

    y_val_flat = np.argmax(y_val, axis=-1).flatten().tolist()
    predictions_flat = predictions.flatten().tolist()

    y_val_flat.extend(range(len(dl.classes)))
    predictions_flat.extend(range(len(dl.classes)))

    cnf_matrix = confusion_matrix(y_val_flat, predictions_flat).tolist()
    for i, row in enumerate(cnf_matrix):
        row.insert(0, dl.classes[i])

    print(classification_report(
        y_val_flat, predictions_flat,
        target_names=dl.classes))

    print(tabulate(cnf_matrix, headers=[c[:1] for c in dl.classes]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument(
        "model_path",
        help="path to model, assuming prediction script generated")
    parser.add_argument("split", help="train/val", choices=['train', 'test'])
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    params = util.get_model_params(args.model_path)
    evaluate(args, params)
