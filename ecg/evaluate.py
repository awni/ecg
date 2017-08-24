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


def parse_classification_report(report):
    lines = report.split('\n')
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: len(lines)]:
        t = re.split('\s\s+', line.strip())
        if len(t) < 2:
            continue
        v = [float(x) for x in t[1: len(t) - 1]]
        if t[0] != 'avg / total':
            class_names.append(t[0])
        support.append(int(t[-1]))
        plotMat.append(v)
    return np.array(plotMat), support, class_names


class Evaluator():
    def __init__(self, scorer):
        self.scorer = scorer
        self.score_params = {}

    def _seq_to_set_gt(self):
        self.set_gt = self._seq_to_set(self.seq_gt)

    def _seq_to_set_preds(self):
        self.set_preds = self._seq_to_set(self.seq_preds)

    def _repeat_seq_preds(self):
        self.seq_preds = np.tile(
            self.seq_preds, (self.seq_gt.shape[0], 1))

    def _flat_seq_gt(self):
        self.seq_gt = self.seq_gt.reshape((-1, self.seq_gt.shape[-1]))

    def score(self, gt, preds):
        self.scorer.score(
            gt,
            preds,
            **self.score_params)

    def evaluate(self, ground_truths, probs, metric='seq'):
        assert(metric in ['set', 'seq'])
        self._to_seq_gt(ground_truths)
        self._to_seq_preds(probs)
        self._repeat_seq_preds()
        self._flat_seq_gt()
        if metric == 'seq':
            self.score(
                self.seq_gt.ravel(), self.seq_preds.ravel())
        else:
            self._seq_to_set_gt()
            self._seq_to_set_preds()
            self.score(
                self.set_gt, self.set_preds)


class MulticlassEval(Evaluator):
    def __init__(self, scorer, classes, decoder=None):
        Evaluator.__init__(self, scorer)
        self.classes = classes
        self.decoder = decoder
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

    def _to_seq_preds(self, probs):
        if self.decoder is not None:
            predictions = np.array(
                [self.decoder.beam_search(probs_indiv)
                 for probs_indiv in tqdm(probs)])
        else:
            predictions = np.argmax(probs, axis=-1)

        self.seq_preds = predictions


class BinaryEval(Evaluator):
    def __init__(self, scorer, class_int, class_name, threshold):
        Evaluator.__init__(self, scorer)
        self.threshold = threshold
        self.class_int = class_int
        self.class_name = class_name
        self.score_params = {
            'class_name': class_name,
            'threshold': threshold
        }

    def _seq_to_set(self, arr):
        set_records = []
        for record_labels in arr.astype('int'):
            unique = set(np.unique(record_labels))
            unique.discard(0)
            set_records.append(list(unique))
        from sklearn import preprocessing
        lb = preprocessing.MultiLabelBinarizer(classes=[1])
        return lb.fit_transform(set_records)

    def _to_seq_gt(self, ground_truths):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == self.class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        self.seq_gt = ground_truths

    def _to_seq_preds(self, probs):
        probs = np.copy(probs)
        predictions = probs[:, :, self.class_int]
        mask_as_one = predictions >= self.threshold
        predictions[mask_as_one] = 1
        predictions[~mask_as_one] = 0
        self.seq_preds = predictions


def evaluate_binary(
        ground_truths, probs, classes, thresholds, metric, model_title,
        plot=False):
    scorer = score.BinaryScorer(model_title=model_title, metric=metric)
    for class_int in tqdm(range(len(classes))):
        for threshold in thresholds:
            evaluator = BinaryEval(
                scorer, class_int, classes[class_int], threshold)
            evaluator.evaluate(ground_truths, probs, metric=metric)
    scorer.display_scores(plot=plot)


def evaluate_multiclass(
        ground_truths, probs, classes, metric, model_title,
        decoder=None, plot=False, display_scores=True):
    scorer = score.MulticlassScorer(metric=metric, model_title=model_title)
    evaluator = MulticlassEval(scorer, classes, decoder=decoder)
    evaluator.evaluate(ground_truths, probs, metric=metric)
    if display_scores:
        scorer.display_scores(plot=plot)
    return evaluator


def evaluate_all(
        gt, probs, classes,
        model_title='', thresholds=[0.5], decoder=None, plot=False):
    for metric in ['seq', 'set']:
        evaluate_multiclass(
            gt, probs, classes, metric, model_title,
            decoder=decoder, plot=plot)
        evaluate_binary(
            gt, probs, classes, thresholds, metric, model_title, plot=plot)


def evaluate(args, params):
    gt, probs, classes = predict.load_predictions(args.prediction_folder)
    thresholds = np.linspace(0, 1, 6, endpoint=False)
    evaluate_all(
        gt, probs, classes, model_title=(',').join(params["model_paths"]),
        thresholds=thresholds, decoder=None, plot=args.plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_folder", help="path to prediction folder")
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    params = json.load(open(args.prediction_folder + '/params.json', 'r'))
    evaluate(args, params)
