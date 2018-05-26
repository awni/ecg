from sklearn import preprocessing
import pywt
import numpy as np
from tqdm import tqdm
import warnings
import scipy.signal as scs

class Normalizer(object):
    def __init__(self, strategy):
        self.scaler = None
        self.strategy = strategy

    def _dim_fix(self, x):
        if (len(x.shape) == 2):
            warnings.warn("Expanding Dimensions...")
            x = np.expand_dims(x, axis=-1)
        assert(len(x.shape) == 3)
        return x

    def fit(self, x):
        print('Fitting Normalization: ' + self.strategy)
        x = self._dim_fix(x)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        if self.strategy == 'standard_scale':
            self.scaler = preprocessing.StandardScaler().fit(x)
        elif self.strategy == 'min_max':
            self.scaler = preprocessing.MinMaxScaler(
                feature_range=(-1, 1)).fit(x)
        elif self.strategy == 'robust_scale':
            self.scaler = preprocessing.RobustScaler().fit(x)
        else:
            raise ValueError("Strategy not found!")

    def transform(self, x):
        print('Applying Normalization...')
        x = self._dim_fix(x)
        original_shape = x.shape
        new_shape = (x.shape[0]*x.shape[1], x.shape[2])
        return self.scaler.transform(
            x.reshape(new_shape)).reshape(original_shape)


class BandPassFilter(object):
    def filt(self, data, lowcut=0.5, highcut=90, fs=200, order=5):
        """
        http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scs.butter(order, [low, high], btype='bandpass')
        y = scs.lfilter(b, a, data)
        return y

    def transform(self, x):
        print('Applying Butterworth Filter...')
        return np.array([self.filt(x_indiv) for x_indiv in x])


class DiscreteWaveletTransformer(object):
    def __init__(self, wavelet_fns, level):
        self.transforms = wavelet_fns
        self.level = level

    def transform(self, x):
        print('Applying Wavelet Transformations...')
        x_new = []
        for x_indiv in tqdm(x):
            x_indiv_trans = []
            for wavefn in self.transforms:
                transform = np.array(pywt.wavedec(
                    x_indiv, wavefn, level=self.level)[:2])
                if(len(x_indiv_trans) > 0 and
                        transform.shape[1] != len(x_indiv_trans[0])):
                    warnings.warn(
                        "Reshaping to proper length after wavelet transform")
                    transform = transform[:, :len(x_indiv_trans[0])]
                x_indiv_trans.extend(transform)
            x_new.append(np.array(x_indiv_trans).T)
        x_new = np.array(x_new)
        return x_new


class ContinuousWaveletTransformer(object):
    def __init__(self, wavelet_fns):
        self.transforms = wavelet_fns
        self.widths = np.linspace(1, 100, 10)  # TODO: parameterize

    def transform(self, x):
        print('Applying Wavelet Transformations...')
        x_new = []
        for x_indiv in tqdm(x):
            x_indiv_trans = []
            for wavefn in self.transforms:
                transform, _ = pywt.cwt(x_indiv, self.widths, 'mexh')
                if(transform.shape[1] != len(x_indiv)):
                    warnings.warn(
                        "Reshaping to proper length after wavelet transform")
                    transform = transform[:, :len(x_indiv)]
                x_indiv_trans.extend(transform)
            x_new.append(np.array(x_indiv_trans).T)
        x_new = np.array(x_new)
        return x_new
