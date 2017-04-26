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
        precision = ppv = 1 if (tp == 0 and fp == 0) else tp / (tp + fp)
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
        try:
            self.plot_precision_recall()
        except:
            print("Cannot plot.")

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
        try:
            self.plot_confusion_matrix()
            self.plot_classification_report()
        except:
            print("Cannot plot.")

    def plot_confusion_matrix(self):
        if self.metric is not 'set':
            plot.plot_confusion_matrix(
                self.cnf,
                self.classes,
                title=self.metric)

    def plot_classification_report(self):
        plot.plot_classification_report(
            self.report,
            title="Classification report (" + self.metric + " F1)")
