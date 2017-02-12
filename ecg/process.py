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
        x_train = np.array(loader.x_train)
        y_train = np.array(loader.y_train)
        x_test = np.array(loader.x_test)
        y_test = np.array(loader.y_test)

        if self.use_bandpass_filter is True:
            bp_filter = featurize.BandPassFilter()
            x_train = bp_filter.transform(x_train)
            x_test = bp_filter.transform(x_test)

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
            x_train = wavelet_transformer.transform(x_train)
            x_test = wavelet_transformer.transform(x_test)

        if self.normalizer is not False:
            if fit is True:
                self.n = featurize.Normalizer(self.normalizer)
                self.n.fit(x_train)
            if len(x_train) > 0:
                x_train = self.n.transform(x_train)
            if len(x_test) > 0:
                x_test = self.n.transform(x_test)

        if self.ignore_classes is not False:
            for ignore_class in self.ignore_classes:
                print("Ignoring class: " + ignore_class)
                for split in ['_train', '_test']:
                    indices = np.where(np.sum(getattr(
                        locals, 'y' + split) == ignore_class, axis=1) == 0)[0]
                    for prop in ['x', 'y']:
                        setattr(self, prop + split, getattr(
                            self, prop + split)[indices])

        y_train = self.transform_to_int_label(y_train, loader)
        y_test = self.transform_to_int_label(y_test, loader)
        return (x_train, y_train, x_test, y_test)

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
