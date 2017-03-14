import evaluate
import json
import load
import numpy as np
import argparse


def get_ground_truths_and_human_prediction(ground_truths, num_reviewers):
    ground_truths = np.swapaxes(ground_truths, 0, 1)
    select_index = np.random.randint(0, num_reviewers, size=len(ground_truths))
    human_prediction_mask = np.zeros(
        (select_index.size, select_index.max()+1), dtype='bool')
    human_prediction_mask[np.arange(
        select_index.size), select_index] = 1

    predictions = ground_truths[human_prediction_mask]

    ground_truths = ground_truths[~human_prediction_mask].reshape(
        (len(ground_truths), num_reviewers - 1, -1))
    ground_truths = np.swapaxes(ground_truths, 0, 1)

    return ground_truths, predictions


def agreement(args, params):
    x, ground_truths, classes = load.load_test(params)
    num_reviewers = params["num_reviewers"]
    ground_truths, predictions = get_ground_truths_and_human_prediction(
        ground_truths, num_reviewers)
    predictions = np.tile(predictions, (num_reviewers - 1, 1))

    evaluate.compute_scores(ground_truths, predictions, classes)


def compute_metrics(ground_truths, predictions, classes):
    all_metrics = []
    for class_int, class_name in enumerate(classes):
        print(class_name)
        ground_truth_class = evaluate.get_ground_truths_for_class(
            ground_truths, class_int)
        preds = evaluate.get_ground_truths_for_class(
            predictions, class_int)
        # Repeat the predictions by the number of reviewers.
        preds = np.tile(preds, (ground_truth_class.shape[0], 1))
        class_metrics = evaluate.compute_scores_class(
            ground_truth_class, preds)
        print(class_metrics)
        all_metrics.append(class_metrics)


def class_agreement(args, params):
    x, ground_truths, classes = load.load_test(params)
    num_reviewers = params["num_reviewers"]
    ground_truths, predictions = get_ground_truths_and_human_prediction(
        ground_truths, num_reviewers)
    compute_metrics(ground_truths, predictions, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    params = json.load(open('configs/test.json', 'r'))
    agreement(args, params)
    class_agreement(args, params)
