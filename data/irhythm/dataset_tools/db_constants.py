"""
Constants to be shared between different iRhythm databases
"""

# ECG file constants
ECG_SAMP_RATE = 200.0  # Hz
ECG_COUNTS_PER_MV = 1049  # counts/mv
ECG_BYTES_PER_SAMPLE = 2


# File extensions
ECG_EXT = '.ecg'
EPI_EXT = '.episodes.json'
BEA_EXT = '.rpeak'


# Rhythm codes
rhythm_code2name = {
    100: 'NSR', 200: 'SVT', 300: 'SUDDEN_BRADY', 400: 'AVB_TYPE2',
    500: 'PAUSE', 600: 'AFIB', 700: 'VT', 800: 'BIGEMINY', 900: 'TRIGEMINY',
    1000: 'VF', 1100: 'PACING', 1999: 'WENCKEBACH', 9999: 'NOISE'
}

rhythm_name2code = {
    'NSR': 100, 'SVT': 200, 'SUDDEN_BRADY': 300, 'AVB_TYPE2': 400,
    'PAUSE': 500, 'AFIB': 600, 'VT': 700, 'BIGEMINY': 800, 'TRIGEMINY': 900,
    'VF': 1000, 'PACING': 1100, 'WENCKEBACH': 1999, 'NOISE': 9999
}

