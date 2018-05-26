from __future__ import division
from __future__ import print_function

import numpy as np
import featurize
import collections

class Processor(object):
    def __init__(
        self,
        use_one_hot_labels=True,
        normalizer=False,
        relabel_classes={},
        ignore_classes=[],
        **kwargs
    ):
        self.normalizer = normalizer
        self.ignore_classes = ignore_classes
        self.relabel_classes = relabel_classes
        self.use_one_hot_labels = use_one_hot_labels
        self.n = None
        self.classes = None
        self.int_to_class = None
        self.class_to_int = None

    def process_x(self, fit):
        if self.normalizer is not False:
            if fit is True and len(self.x_train) > 0:
                self.n = featurize.Normalizer(self.normalizer)
                self.n.fit(self.x_train)
                if len(self.x_train) > 0:
                    self.x_train = self.n.transform(self.x_train)
            if len(self.x_test) > 0:
                self.x_test = self.n.transform(self.x_test)

    def process_y(self):
        self.y_train = self.transform_to_int_label(self.y_train)
        self.y_test = self.transform_to_int_label(self.y_test)

    def process(self, loader, fit=True):
        self.setup_label_mappings(loader, fit)
        self.x_train = np.array(loader.x_train)
        self.x_test = np.array(loader.x_test)
        self.process_x(fit)

        self.y_train = np.array(loader.y_train)
        self.y_test = np.array(loader.y_test)
        self.process_y()

        return (self.x_train, self.y_train, self.x_test, self.y_test)

    def setup_label_mappings(self, loader, fit):
        if len(self.relabel_classes) > 0:
            print("Relabelling Classes...")
            for split in ['_train', '_test']:
                y = getattr(loader, 'y' + split)
                y_new = [[self.relabel_classes[s] if s in self.relabel_classes
                         else s for s in y_indiv] for y_indiv in y]
                setattr(loader, 'y' + split, y_new)

        if len(self.ignore_classes) > 0:
            for ignore_class in self.ignore_classes:
                print("Ignoring class: " + ignore_class)
                for split in ['_train', '_test']:
                    attr = getattr(loader, 'y' + split)
                    if len(attr) > 0:
                        indices = np.where(np.sum(attr == ignore_class,
                                           axis=1) == 0)[0]
                        for prop in ['x', 'y']:
                            setattr(loader, prop + split, getattr(
                                loader, prop + split)[indices])

        if fit is True:
            y_tot = list(loader.y_train) + list(loader.y_test)
            label_counter = collections.Counter(
                l for labels in y_tot for l in labels)
            self.classes = sorted([c for c, _ in label_counter.most_common()])
            self.int_to_class = dict(
                zip(range(len(self.classes)), self.classes))
            self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def transform_to_int_label(self, y_split):
        labels_mod = []
        for label in y_split:
            label_mod = np.array([self.class_to_int[c] for c in label])
            if self.use_one_hot_labels is True:
                tmp = np.zeros((len(label_mod), len(self.int_to_class)))
                tmp[np.arange(len(label_mod)), label_mod] = 1
                label_mod = tmp
            labels_mod.append(label_mod)
        return np.array(labels_mod)
