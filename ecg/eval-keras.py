import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from loader import Loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("prediction_path", help="path to prediction pickle")
    parser.add_argument("--refresh", help="whether to refresh cache")
    args = parser.parse_args()

    dl = Loader(
        args.data_path,
        use_one_hot_labels=True,
        seed=2016,
        use_cached_if_available=not args.refresh)

    x_val = dl.x_test[:, :, np.newaxis]
    y_val = dl.y_test
    print("Validation size: " + str(len(x_val)) + " examples.")

    predictions = np.load(open(args.prediction_path, 'rb'))

    y_val_flat = np.argmax(y_val, axis=-1).flatten().tolist()
    predictions_flat = np.argmax(predictions, axis=-1).flatten().tolist()

    cnf_matrix = confusion_matrix(y_val_flat, predictions_flat).tolist()
    for i, row in enumerate(cnf_matrix):
        row.insert(0, dl.classes[i])

    y_val_flat.extend(range(len(dl.classes)))
    predictions_flat.extend(range(len(dl.classes)))

    print(classification_report(
        y_val_flat, predictions_flat,
        target_names=dl.classes))

    print(tabulate(cnf_matrix, headers=[c[:1] for c in dl.classes]))
