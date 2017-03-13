import evaluate
import json
import load
import numpy as np
import argparse


def agreement(args, params):
    x, ground_truths, classes = load.load_test(params)

    num_reviewers = params["num_reviewers"]

    select_index = np.random.randint(0, num_reviewers, size=len(ground_truths))
    human_prediction_mask = np.zeros(
        (select_index.size, select_index.max()+1), dtype='bool')
    human_prediction_mask[np.arange(
        select_index.size), select_index] = 1

    predictions = ground_truths[human_prediction_mask]
    new_ground_truths = ground_truths[~human_prediction_mask].reshape(
        (len(ground_truths), num_reviewers - 1, -1))

    predictions = np.tile(predictions, (num_reviewers - 1, 1))
    new_ground_truths = np.swapaxes(new_ground_truths, 0, 1)

    evaluate.compute_scores(new_ground_truths, predictions, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    params = json.load(open('configs/test.json', 'r'))
    agreement(args, params)
