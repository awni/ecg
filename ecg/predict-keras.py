from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import numpy as np
from keras.models import load_model

from loader import Loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("model_path", help="path to model")
    parser.add_argument("--refresh", help="whether to refresh cache", action="store_true")
    args = parser.parse_args()

    dl = Loader(
        args.data_path,
        use_one_hot_labels=True,
        seed=2016,
        use_cached_if_available=not args.refresh)

    x_val = dl.x_test[:, :, np.newaxis]
    y_val = dl.y_test
    print("Validation size: " + str(len(x_val)) + " examples.")

    model = load_model(args.model_path)
    predictions = model.predict(x_val)
    with open(args.model_path + '-preds.pkl', 'wb') as outfile:
        np.save(outfile, predictions)
