import argparse
import numpy as np
from keras.models import load_model

from loader import Loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("model_path", help="path to model")
    parser.add_argument("--refresh", help="whether to refresh cache")
    args = parser.parse_args()

    dl = Loader(
        args.data_path,
        use_one_hot_labels=True,
        use_cached_if_available=not args.refresh)

    x_val = dl.x_test[:, :, np.newaxis]
    y_val = dl.y_test
    print("Validation size: " + str(len(x_val)) + " examples.")

    model = load_model(args.model_path)
    predictions = model.predict(x_val)
    with open(args.model_path + '-preds.pkl', 'wb') as outfile:
        np.save(outfile, predictions)
