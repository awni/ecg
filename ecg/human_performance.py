import evaluate
import json
import load
import argparse
import numpy as np

NUM_TEST_REVIEWERS = 6


def human_performance(args, params):
    _, ground_truths, processor, dl = load.load_test(params)
    gt_all = []
    probs_all = []
    for i in range(NUM_TEST_REVIEWERS):
        params["epi_ext"] = "_rev" + str(i) + ".episodes.json"
        # TODO: check x's are equal
        _, probs, dl = load.load_x_y_with_processor(params, processor)
        evaluate.evaluate_all(
            ground_truths, probs, processor.classes,
            model_title='Human Performance with review ' + str(i))
        gt_all.append(ground_truths)
        probs_all.append(probs)
    ground_truths = np.concatenate(tuple(gt_all), axis=1)
    probs = np.concatenate(tuple(probs_all), axis=0)
    evaluate.evaluate_all(
            ground_truths, probs, processor.classes,
            model_title='Human Performance Average')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    params = json.load(open('configs/test.json', 'r'))
    human_performance(args, params)
