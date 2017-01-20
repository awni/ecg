from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

from loader import Loader
import decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("model_path", help="path to model, assuming prediction script generated")
    parser.add_argument("split", help="train/val", choices=['train', 'test'])
    parser.add_argument('--decode', action='store_true')
    parser.add_argument("--refresh", help="whether to refresh cache", action="store_true")
    args = parser.parse_args()

    dl = Loader(
        args.data_path,
        use_one_hot_labels=True,
        seed=2016,
        use_cached_if_available=not args.refresh)

    if args.split == 'train':
        x_val = dl.x_train[:, :, np.newaxis]
        y_val = dl.y_train
    else:
        x_val = dl.x_test[:, :, np.newaxis]
        y_val = dl.y_test

    print("Size: " + str(len(x_val)) + " examples.")

    predictions = np.load(open(args.model_path + '-pred-' + args.split + '.pkl', 'rb'))

    if args.decode is True:
        language_model = decoder.LM(dl.y_train, dl.output_dim, order=2)
        predictions = np.array([decoder.beam_search(prediction, language_model)
                                for prediction in tqdm(predictions)])
    else:
        predictions = np.argmax(predictions, axis=-1)

    y_val_flat = np.argmax(y_val, axis=-1).flatten().tolist()
    predictions_flat = predictions.flatten().tolist()

    cnf_matrix = confusion_matrix(y_val_flat, predictions_flat).tolist()
    for i, row in enumerate(cnf_matrix):
        row.insert(0, dl.classes[i])

    y_val_flat.extend(range(len(dl.classes)))
    predictions_flat.extend(range(len(dl.classes)))

    print(classification_report(
        y_val_flat, predictions_flat,
        target_names=dl.classes))

    print(tabulate(cnf_matrix, headers=[c[:1] for c in dl.classes]))
