from sklearn import preprocessing
import pywt
import numpy as np
from builtins import zip


class Normalizer(object):
    def __init__(self):
        self.scaler = None

    def _dim_fix(self, x):
        if (len(x.shape) == 2):
            x = np.expand_dims(x, axis=-1)
        assert(len(x.shape) == 3)
        return x

    def fit(self, x):
        x = self._dim_fix(x)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        self.scaler = preprocessing.StandardScaler().fit(x)

    def transform(self, x):
        x = self._dim_fix(x)
        original_shape = x.shape
        new_shape = (x.shape[0]*x.shape[1], x.shape[2])
        return self.scaler.transform(
            x.reshape(new_shape)).reshape(original_shape)


class WaveletTransformer(object):
    def __init__(self, wavelet_type='db1'):
        self.wavelet_type = wavelet_type

    def fit(self, x):
        pass

    def transform(self, x):
        x_new = np.array([np.array(pywt.dwt(x_indiv, self.wavelet_type)).T for x_indiv in x])
        return x_new
