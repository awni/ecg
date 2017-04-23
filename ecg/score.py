from __future__ import print_function
from __future__ import division
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import plot


class Scorer(object):
    def __init__(self, model_title='', metric=''):
        self.metric = metric
        self.model_title = model_title

    def compute_confusion_matrix(self, gt, preds):
        def flatten_gt_and_preds(gt, preds):
            def flatten_and_to_list(inp):
                return inp.flatten().tolist()
            return flatten_and_to_list(gt), flatten_and_to_list(preds)

        self.cnf = confusion_matrix(*flatten_gt_and_preds(gt, preds)).tolist()

    def display_scores(self):
        print()
        print('===Metric==', self.metric)
        print('===Model===', self.model_title)
        print()


class BinaryScorer(Scorer):
    def __init__(self, **params):
        Scorer.__init__(self, **params)
        self.rows = []
        self.rows_by_classes = defaultdict(list)
        self.headers = [
            'c_name',
            'specificity',
            'precision',
            'recall',
            'f1',
            'threshold'
        ]

    def score(self, gt, preds, class_name=None, threshold=None):
        self.compute_confusion_matrix(gt, preds)

        tn, fp = self.cnf[0][0], self.cnf[0][1]
        fn, tp = self.cnf[1][0], self.cnf[1][1]

        recall = sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        try:
            precision = ppv = tp / (tp + fp)
        except ZeroDivisionError:
            precision = ppv = 0
        f1 = (2 * precision * recall) / (precision + recall)

        row = [
            class_name,
            specificity,
            ppv,
            sensitivity,
            f1,
            threshold
        ]
        self.rows_by_classes[class_name].append(row)
        self.rows.append(row)

    def display_scores(self):
        Scorer.display_scores(self)
        print(tabulate(self.rows, headers=self.headers, floatfmt=".3f"))
        self.plot_precision_recall()

    def plot_precision_recall(self):
        plot.plot_precision_recall(
            classes=self.rows_by_classes.keys(),
            class_data=self.rows_by_classes,
            metric=self.metric)


class MulticlassScorer(Scorer):
    def score(self, gt, preds, classes=None):
        self.compute_confusion_matrix(gt, preds)
        self.classes = classes
        self.report = classification_report(
                gt, preds, target_names=self.classes, digits=3)

    def display_scores(self):
        Scorer.display_scores(self)
        print(self.report)
