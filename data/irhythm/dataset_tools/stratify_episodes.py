#!/usr/bin/env python
"""
Creates train/test/(val) set splits for a given ECG dataset.

Usage:
 startify_episodes.py  <db_file> --rng=<rng_seed> --train=<train_frac> --test=<test_frac>
                       [--method=<strat_method>] [--exclude=<rhy_names>]

Arguments:
<db_file>                    Path to pickle file containing episodes' info

Options:
--rng=<rng_seed>             Random generator seed
--train=<train_frac>         Fraction of dataset to assign to the training set.
--test=<test_frac>           Fraction of dataset to assign to the testing set.
--method=<strat_method>      Data stratification method to be performed,
                             (currently only one method is implemented) [default: sub_rhy]
--exclude=<rhy_names>        List of rhythm names (separated by comma) to be
                             excluded for stratification.
-h --help                    Show this documentation.
"""

import numpy as np
import cPickle as pickle
from docopt import docopt


def _strat_over_subject_rhythms(db_info, train_frac, test_frac,
                                exclude_list):
    """
    Assigns episodes in the input dataframe to one of train, test, and val sets.
    In this stratification method, all instances of a rhythm-type belonging to
    the same record will be in one set (train, test, or val) only.
    :param db_info: dataframe containing episodes' info (with rec, start_idx,
                    end_idx and rhythm_name columns)
    :param train_frac: fraction of data to be assigned for training
    :param test_frac: fraction of data to be assigned for testing
    :param exclude_list: list of rhythm names to be excluded from training
    """

    db_info['set'] = np.nan
    db_groups = db_info.groupby('rhythm_name')

    for rhy_name, rhy_group in db_groups:
        if rhy_name not in exclude_list:
            records = np.random.permutation(rhy_group.rec.unique())
            n_records = len(records)

            if n_records < 4:
                print 'Warning: only %d record(s) contain %s' % (n_records, rhy_name)

            n_train = int(train_frac * n_records)
            n_test = int(test_frac * n_records)

            db_info.loc[(db_info.rhythm_name == rhy_name) &
                        (db_info.rec.isin(records[:n_train])),
                        'set'] = 'train'

            db_info.loc[(db_info.rhythm_name == rhy_name) &
                        (db_info.rec.isin(records[n_train:n_train+n_test])),
                        'set'] = 'test'

            db_info.loc[(db_info.rhythm_name == rhy_name) &
                        (db_info.rec.isin(records[n_train+n_test:])),
                        'set'] = 'val'


if __name__ == '__main__':
    args = docopt(__doc__)

    db_file = args['<db_file>']
    rng_seed = int(args['--rng'])
    strat_method = args['--method']
    train_frac = float(args['--train'])
    test_frac = float(args['--test'])
    exclude = args['--exclude']

    exclude_list = []
    if exclude:
        exclude_list = exclude.split(',')
        exclude_list = [rhy_name.strip().upper() for
                        rhy_name in exclude_list]

    assert train_frac + test_frac <= 1, 'The sum of train and test fractions ' \
                                        'must be <= 1'

    # set random seed
    np.random.seed(rng_seed)
    print('Set random number generator seed: %d' % rng_seed)

    with open(db_file, 'rb') as f:
        db_info = pickle.load(f)

    if strat_method == 'sub_rhy':
        _strat_over_subject_rhythms(db_info, train_frac, test_frac,
                                    exclude_list)
    else:
        raise ValueError('Requested method is not implemented yet!')

    with open(db_file, 'wb') as f:
        pickle.dump(db_info, f)

    print 'Stratification completed!'
    print 'Input pkl file is updated to include stratification info.'



