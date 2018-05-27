import evaluate
import json
import load
import argparse
import numpy as np

TEST_REVIEWS = [0, 1, 2, 3, 4, 5]

def get_matching_indices(x_gt, x_rev):
    gt_i, rev_i = [], []
    for index, x_i in enumerate(x_rev):
        matching = np.nonzero((x_gt == x_i).all(axis=1))[0]
        if len(matching) == 1:
            rev_i.append(index)
            match = matching[0]
            gt_i.append(match)
    return gt_i, rev_i

def human_gt_and_probs(params, x_gt, ground_truths, processor, review_indiv=False):
    gt_all = []
    probs_all = []
    for i in TEST_REVIEWS:
        params["epi_ext"] = "_rev" + str(i) + ".episodes.json"
        x_rev, probs, dl = load.load_x_y_with_processor(params, processor)
        gt_i, rev_i = get_matching_indices(x_gt, x_rev)
        gt = ground_truths[:, gt_i]
        probs = probs[rev_i]
        if review_indiv is True:
            evaluate.evaluate_all(
                gt, probs, processor.classes,
                model_title='Human Performance with review ' + str(i))
        gt_all.append(gt)
        probs_all.append(probs)
    ground_truths = np.concatenate(tuple(gt_all), axis=1)
    probs = np.concatenate(tuple(probs_all), axis=0)
    return ground_truths, probs

def human_performance(args, params):
    x_gt, ground_truths, processor, dl = load.load_test(
        params, fit_processor=True)
    ground_truths, probs = human_gt_and_probs(
        params, x_gt, ground_truths, processor, review_indiv=True)
    evaluate.evaluate_all(
            ground_truths, probs, processor.classes,
            model_title='Human Performance Average')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_config_file", help="path to config file")
    args = parser.parse_args()
    params = json.load(open(args.test_config_file, 'r'))
    human_performance(args, params)
