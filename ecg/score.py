from __future__ import division
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate


class Scorer(object):
    def compute_confusion_matrix(self, gt, preds):
        def flatten_gt_and_preds(gt, preds):
            def flatten_and_to_list(inp):
                return inp.flatten().tolist()
            return flatten_and_to_list(gt), flatten_and_to_list(preds)

        self.cnf = confusion_matrix(*flatten_gt_and_preds(gt, preds)).tolist()


class BinaryScorer(Scorer):
    def __init__(self):
        self.rows = []
        self.headers = [
            'c_name',
            'threshold',
            'specificity',
            'precision',
            'recall',
            'f1'
        ]

    def score(
            self, gt, preds, classes,
            class_name=None, threshold=None, **params):

        self.compute_confusion_matrix(gt, preds)

        tn, fp = self.cnf[0][0], self.cnf[0][1]
        fn, tp = self.cnf[1][0], self.cnf[1][1]

        recall = sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = ppv = tp / (tp + fp)
        f1 = (2 * precision * recall) / (precision + recall)

        row = [
            class_name,
            threshold,
            specificity,
            ppv,
            sensitivity,
            f1
        ]
        self.rows.append(row)

    def display_scores(self):
        print(tabulate(self.rows, headers=self.headers))


class MulticlassScorer(Scorer):
    def score(self, gt, preds, classes):
        self.compute_confusion_matrix(gt, preds)
        self.classes = classes
        self.report = classification_report(
                gt, preds, target_names=self.classes, digits=3)

    def display_scores(self, confusion_table=False):
        if confusion_table is True:
            for i, row in enumerate(self.cnf):
                row.insert(0, self.classes[i])

            print(tabulate(self.cnf, headers=[c[:1] for c in self.classes]))

        print(self.report)
