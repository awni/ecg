from __future__ import division
from __future__ import print_function
import numpy as np
import featurize


class Processor(object):
    def __init__(
        self,
        use_one_hot_labels=True,
        normalizer=False,
        ignore_classes=[],
        wavelet_fns=[],
        wavelet_type='discrete',
        wavelet_level=1,
        use_bandpass_filter=False,
        **kwargs
    ):
        self.wavelet_fns = wavelet_fns
        self.normalizer = normalizer
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.use_one_hot_labels = use_one_hot_labels
        self.ignore_classes = ignore_classes
        self.use_bandpass_filter = use_bandpass_filter
        self.n = None

    def process(self, loader, fit=True):
        self.x_train = np.array(loader.x_train)
        self.y_train = np.array(loader.y_train)
        self.x_test = np.array(loader.x_test)
        self.y_test = np.array(loader.y_test)

        if self.use_bandpass_filter is True:
            bp_filter = featurize.BandPassFilter()
            self.x_train = bp_filter.transform(self.x_train)
            self.x_test = bp_filter.transform(self.x_test)

        if len(self.wavelet_fns) != 0:
            if (self.wavelet_type == 'discrete'):
                wavelet_transformer = \
                    featurize.DiscreteWaveletTransformer(
                        self.wavelet_fns, self.wavelet_level)
            elif (self.wavelet_type == 'continuous'):
                wavelet_transformer = \
                    featurize.ContinuousWaveletTransformer(self.wavelet_fns)
            else:
                raise ValueError("Wavelet type not defined.")
            self.x_train = wavelet_transformer.transform(self.x_train)
            self.x_test = wavelet_transformer.transform(self.x_test)

        if self.normalizer is not False:
            if fit is True:
                self.n = featurize.Normalizer(self.normalizer)
                self.n.fit(self.x_train)
            if len(self.x_train) > 0:
                self.x_train = self.n.transform(self.x_train)
            if len(self.x_test) > 0:
                self.x_test = self.n.transform(self.x_test)

        if self.ignore_classes is not False:
            for ignore_class in self.ignore_classes:
                print("Ignoring class: " + ignore_class)
                for split in ['_train', '_test']:
                    attr = getattr(self, 'y' + split)
                    if len(attr) > 0:
                        indices = np.where(np.sum(attr == ignore_class,
                                           axis=1) == 0)[0]
                        print(indices)
                        for prop in ['x', 'y']:
                            setattr(self, prop + split, getattr(
                                self, prop + split)[indices])
        self.y_train = self.transform_to_int_label(self.y_train, loader)
        self.y_test = self.transform_to_int_label(self.y_test, loader)
        return (self.x_train, self.y_train, self.x_test, self.y_test)

    def transform_to_int_label(self, y_split, loader):
        labels_mod = []
        for label in y_split:
            label_mod = np.array([loader.class_to_int[c] for c in label])
            if self.use_one_hot_labels is True:
                tmp = np.zeros((len(label_mod), len(loader.int_to_class)))
                tmp[np.arange(len(label_mod)), label_mod] = 1
                label_mod = tmp
            labels_mod.append(label_mod)
        return np.array(labels_mod)
