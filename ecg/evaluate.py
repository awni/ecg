from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from tqdm import tqdm
import load
import json
import decode
import util
import predict
import score


def get_class_gt_and_preds(
        ground_truths,
        probs,
        class_int,
        threshold):

    def get_binary_preds_for_class(probs, class_int, threshold):
        probs = np.copy(probs)
        class_probs = probs[:, :, class_int]
        mask_as_one = class_probs >= threshold
        class_probs[mask_as_one] = 1
        class_probs[~mask_as_one] = 0
        return class_probs

    def get_ground_truths_for_class(ground_truths, class_int):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        return ground_truths

    ground_truth_class = get_ground_truths_for_class(
        ground_truths, class_int)
    predictions = get_binary_preds_for_class(
        probs, class_int, threshold)
    predictions = np.tile(
        predictions, (ground_truth_class.shape[0], 1))
    return ground_truth_class, predictions


def get_aggregate_gt_and_preds(
        ground_truths, probs, decoder=False, language_model=None):

    def decode_probs(probs, decoder, language_model):
        if decoder is True and language_model is not None:
            raise NotImplementedError()  # TODO: fix
            predictions = np.array(
                [decode.beam_search(probs_indiv, language_model)
                 for probs_indiv in tqdm(probs)])
        else:
            predictions = np.argmax(probs, axis=-1)
        return predictions

    predictions = decode_probs(probs, decoder, language_model)
    predictions = np.tile(
        predictions, (ground_truths.shape[0], 1))
    return predictions


def evaluate_classes(ground_truths, probs, classes, thresholds=[0.5]):
    def evaluate_class(ground_truths, probs, classes, class_int, threshold):
        ground_truth_class, predictions = get_class_gt_and_preds(
            ground_truths, probs, class_int, threshold)
        classes_binarized = ['None', classes[class_int]]
        scores = score.score(
            ground_truth_class,
            predictions,
            classes_binarized,
            binary_evaluate=True)
        print(classes[class_int], scores, threshold)

    for class_int in range(len(classes)):
        for threshold in thresholds:
            evaluate_class(ground_truths, probs, classes, class_int, threshold)


def evaluate_aggregate(ground_truths, probs, classes, decoder=False):
    predictions = get_aggregate_gt_and_preds(
        ground_truths, probs, decoder=decoder)
    score.score(
        ground_truths,
        predictions,
        classes,
        confusion_table=True,
        report=True)


def evaluate(args, train_params, test_params):
    x, ground_truths, classes = load.load_test(
            test_params,
            train_params=train_params,
            split=args.split)
    probs = predict.get_ensemble_pred_probs(args.model_paths, x)
    evaluate_aggregate(ground_truths, probs, classes, decoder=args.decode)
    evaluate_classes(ground_truths, probs, classes, np.linspace(0, 1, 5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_config_file", help="path to config file")
    parser.add_argument(
        'model_paths',
        nargs='+',
        help="path to models")
    parser.add_argument("--split", help="train/val", choices=['train', 'test'],
                        default='test')
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    train_params = util.get_model_params(args.model_paths[0])
    test_params = train_params.copy()
    test_new_params = json.load(open(args.test_config_file, 'r'))
    test_params.update(test_new_params)
    if "label_review" in test_new_params["EVAL_PATH"]:
        assert(args.split == 'test')
    evaluate(args, train_params, test_params)
