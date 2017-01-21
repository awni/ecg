from sklearn import preprocessing


class Normalizer(object):
    def __init__(self):
        self.scaler = None

    def fit(self, x):
        self.scaler = preprocessing.StandardScaler().fit(x)

    def transform(self, x):
        return self.scaler.transform(x)
