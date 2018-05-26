from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import warnings

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
