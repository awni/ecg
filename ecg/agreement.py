import evaluate
import json
import load
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse


def get_ground_truths_and_human_probs(ground_truths, num_reviewers):
    ground_truths = np.swapaxes(ground_truths, 0, 1)
    select_index = np.random.randint(0, num_reviewers, size=len(ground_truths))
    human_prediction_mask = np.zeros(
        (select_index.size, select_index.max()+1), dtype='bool')
    human_prediction_mask[np.arange(
        select_index.size), select_index] = 1

    predictions = ground_truths[human_prediction_mask]

    enc = OneHotEncoder(sparse=False)
    probs = enc.fit_transform(predictions.reshape(-1, 1)).reshape(
        (predictions.shape[0], predictions.shape[1], -1))

    ground_truths = ground_truths[~human_prediction_mask].reshape(
        (len(ground_truths), num_reviewers - 1, -1))
    ground_truths = np.swapaxes(ground_truths, 0, 1)

    return ground_truths, probs


def agreement_aggregate(ground_truths, probs, classes):
    return evaluate.evaluate_aggregate(ground_truths, probs, classes)


def agreement_classes(ground_truths, probs, classes):
    return evaluate.evaluate_classes(ground_truths, probs, classes)


def agreement(args, params):
    _, ground_truths, classes = load.load_test(params)
    ground_truths, probs = get_ground_truths_and_human_probs(
        ground_truths, params["num_reviewers"])
    agreement_aggregate(ground_truths, probs, classes)
    agreement_classes(ground_truths, probs, classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    params = json.load(open('configs/test.json', 'r'))
    agreement(args, params)
