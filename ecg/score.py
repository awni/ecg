from __future__ import print_function
from __future__ import division
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tabulate import tabulate
import numpy as np
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
        self.rows_by_classes = {}
        self.headers = [
            'c_name',
            'roc_auc',
            'false_pos_rates',
            'true_pos_rates'
        ]

    def score(self, gt, probs, class_name=None):
        auc = roc_auc_score(gt, probs)
        fprs, tprs, _ = roc_curve(gt, probs)
        row = [
            class_name,
            auc,
            fprs,
            tprs
        ]
        self.rows_by_classes[class_name] = row
        self.rows.append(row)

    def display_scores(self, plot_flag=False):
        Scorer.display_scores(self)
        print(tabulate([row[:2] for row in self.rows], headers=self.headers, floatfmt=".3f"))
        if plot_flag is True:
            try:
                plot.plot_roc(
                    classes=self.rows_by_classes.keys(),
                    class_data=self.rows_by_classes,
                    metric=self.metric)
            except:
                print("Cannot plot.")

class MulticlassScorer(Scorer):
    def score(self, gt, probs, classes=None):
        preds = probs
        self.compute_confusion_matrix(gt, preds)
        self.classes = classes
        self.report = classification_report(
                gt, preds, target_names=self.classes, digits=3)

    def display_scores(self, plot_flag=False):
        Scorer.display_scores(self)
        print(self.report)
        if plot_flag is True:
            try:
                if self.metric is not 'set':
                    plot.plot_confusion_matrix(
                        self.cnf,
                        self.classes,
                        title=self.metric)
            except:
                print("Cannot plot.")
