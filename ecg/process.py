from __future__ import division
from __future__ import print_function
import numpy as np
import featurize


class Processor(object):
    def __init__(
        self,
        use_one_hot_labels=True,
        normalizer=False,
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
        self.use_bandpass_filter = use_bandpass_filter
        self.n = None

    def process_x(self, fit):
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

    def process_y(self, loader):
        self.y_train = self.transform_to_int_label(self.y_train, loader)
        self.y_test = self.transform_to_int_label(self.y_test, loader)

    def process(self, loader, fit=True):
        self.x_train = np.array(loader.x_train)
        self.x_test = np.array(loader.x_test)
        self.process_x(fit)

        self.y_train = np.array(loader.y_train)
        self.y_test = np.array(loader.y_test)
        self.process_y(loader)

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
