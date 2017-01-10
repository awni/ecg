"""
Functions for transforming or augmenting ECG signals.
"""

import random
import scipy.signal as scs

def rand_flip(ecg, flip_prob=0.5):
    if random.random() < flip_prob:
        return -ecg
    return ecg

def bandpass(data, lowcut=0.5, highcut=90, fs=200, order=5):
    """
    http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scs.butter(order, [low, high], btype='bandpass')
    y = scs.lfilter(b, a, data)
    return y

