import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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

    from keras.models import load_model
    model = load_model(args.model_path)
    predictions = model.predict(x_val)

    y_val_flat = np.argmax(y_val, axis=-1).flatten().tolist()
    y_val_flat.extend(range(len(dl.classes)))
    predictions_flat = np.argmax(predictions, axis=-1).flatten().tolist()
    predictions_flat.extend(range(len(dl.classes)))

    print(classification_report(y_val_flat, predictions_flat, target_names=dl.classes))
