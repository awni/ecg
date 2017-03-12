from __future__ import print_function
from builtins import str
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm
import load
import json
import decode
import util
from joblib import Memory
memory = Memory(cachedir='./cache')


def plot_confusion_matrix(cm, classes, model_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if model_path is not None:
        plt.savefig(util.get_confusion_figure_path(model_path))


@memory.cache
def get_ensemble_pred_probs(model_paths, x):
    def get_model_pred_probs(model_path, x):
        from keras.models import load_model
        model = load_model(model_path)
        probs = model.predict(x, verbose=1)
        return probs

    all_model_probs = [get_model_pred_probs(model_path, x)
                       for model_path in args.model_paths]
    probs = np.mean(all_model_probs, axis=0)
    return probs


def compute_scores(
        ground_truth,
        predictions,
        classes,
        confusion_table=True,
        report=True,
        plot=True):
    ground_truth_flat = ground_truth.flatten().tolist()
    predictions_flat = predictions.flatten().tolist()

    cnf_matrix = confusion_matrix(ground_truth_flat, predictions_flat).tolist()

    if plot is True:
        try:
            plot_confusion_matrix(
                np.log10(np.array(cnf_matrix) + 1),
                classes,
                args.model_path)
        except:
            print("Skipping plot")

    if confusion_table is True:
        for i, row in enumerate(cnf_matrix):
            row.insert(0, classes[i])

        print(tabulate(cnf_matrix, headers=[c[:1] for c in classes]))

    if report is True:
        print(classification_report(
            ground_truth_flat, predictions_flat,
            target_names=classes, digits=3))


def evaluate(args, train_params, test_params, num_reviewers=3):
    x, ground_truths, classes = load.load_test(
        test_params,
        train_params=train_params,
        split=args.split)

    ground_truths = np.swapaxes(ground_truths, 0, 1)

    print("Predicting on:", args.split)

    print("Averaging " + str(len(args.model_paths)) + " model predictions...")
    probs = get_ensemble_pred_probs(args.model_paths, x)

    if args.decode is True:
        raise NotImplementedError()  # TODO: fix
        """
        language_model = decode.LM(dl.y_train, dl.output_dim, order=2)
        predictions = np.array([decode.beam_search(prediction, language_model)
                                for prediction in tqdm(probs)])
        """
    else:
        predictions = np.argmax(probs, axis=-1)

    # Repeat the predictions by the number of reviewers.
    predictions = np.tile(predictions, (num_reviewers, 1))

    compute_scores(ground_truths, predictions, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_config_file", help="path to config file")
    parser.add_argument("--split", help="train/val", choices=['train', 'test'],
                        default='test')
    parser.add_argument(
        'model_paths',
        nargs='+',
        help="path to models")
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    train_params = util.get_model_params(args.model_paths[0])
    test_params = train_params.copy()
    test_new_params = json.load(open(args.test_config_file, 'r'))
    test_params.update(test_new_params)
    if "label_review" in test_new_params["EVAL_PATH"]:
        assert(args.split == 'test')
    evaluate(args, train_params, test_params)
