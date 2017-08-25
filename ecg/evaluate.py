from __future__ import division
from __future__ import print_function
import argparse
import re
import numpy as np
from tqdm import tqdm
import load
import json
import util
import predict
import score
import decode

class Evaluator():
    def __init__(self, scorer):
        self.scorer = scorer
        self.score_params = {}

    def _seq_to_set_gt(self):
        self.set_gt = self._seq_to_set(self.seq_gt)

    def _seq_to_set_probs(self):
        self.set_probs = self._seq_to_set(self.seq_probs)

    def _repeat_seq_probs(self):
        self.seq_probs = np.tile(
            self.seq_probs, (self.seq_gt.shape[0], 1))

    def _flat_seq_gt(self):
        self.seq_gt = self.seq_gt.reshape((-1, self.seq_gt.shape[-1]))

    def score(self, gt, probs):
        self.scorer.score(
            gt,
            probs,
            **self.score_params)

    def evaluate(self, ground_truths, probs, metric='seq'):
        assert(metric in ['set', 'seq'])
        self._to_seq_gt(ground_truths)
        self._to_seq_probs(probs)
        self._repeat_seq_probs()
        self._flat_seq_gt()
        if metric == 'seq':
            self.score(
                self.seq_gt.ravel(),
                self.seq_probs.ravel())
        else:
            self._seq_to_set_gt()
            self._seq_to_set_probs()
            self.score(
                self.set_gt, self.set_probs)


class MulticlassEval(Evaluator):
    def __init__(self, scorer, classes):
        Evaluator.__init__(self, scorer)
        self.classes = classes
        self.score_params = {
            'classes': classes
        }

    def _seq_to_set(self, arr):
        labels = [set(
            np.unique(record_labels).tolist()) for record_labels in arr]
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(labels)

    def _to_seq_gt(self, ground_truths):
        self.seq_gt = ground_truths

    def _to_seq_probs(self, probs):
        self.seq_probs = np.argmax(probs, axis=-1)


class BinaryEval(Evaluator):
    def __init__(self, scorer, class_int, class_name):
        Evaluator.__init__(self, scorer)
        self.class_int = class_int
        self.class_name = class_name
        self.score_params = {
            'class_name': class_name
        }

    def _seq_to_set(self, arr):
        return np.max(arr, axis=-1)

    def _to_seq_gt(self, ground_truths):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == self.class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        self.seq_gt = ground_truths

    def _to_seq_probs(self, probs):
        self.seq_probs = probs[:, :, self.class_int]


def evaluate_binary(
        ground_truths, probs, classes, metric, model_title,
        plot_flag=False):
    scorer = score.BinaryScorer(model_title=model_title, metric=metric)
    for class_int in tqdm(range(len(classes))):
        evaluator = BinaryEval(
            scorer, class_int, classes[class_int])
        evaluator.evaluate(ground_truths, probs, metric=metric)
    scorer.display_scores(plot_flag=plot_flag)


def evaluate_multiclass(
        ground_truths, probs, classes, metric, model_title,
        plot_flag=False, display_scores=True):
    scorer = score.MulticlassScorer(metric=metric, model_title=model_title)
    evaluator = MulticlassEval(scorer, classes)
    evaluator.evaluate(ground_truths, probs, metric=metric)
    if display_scores:
        scorer.display_scores(plot_flag=plot_flag)
    return evaluator


def evaluate_all(
        gt, probs, classes, model_title='', plot_flag=False):
    for metric in ['seq', 'set']:
        evaluate_multiclass(
            gt, probs, classes, metric, model_title, plot_flag=plot_flag)
        evaluate_binary(
            gt, probs, classes, metric, model_title, plot_flag=plot_flag)


def evaluate(args, params):
    gt, probs, classes, _ = predict.load_predictions(args.prediction_folder)
    evaluate_all(
        gt, probs, classes, model_title=(',').join(params["model_paths"]),
        plot_flag=args.plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_folder", help="path to prediction folder")
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    params = json.load(open(args.prediction_folder + '/params.json', 'r'))
    evaluate(args, params)
