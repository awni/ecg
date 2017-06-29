from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import open
from builtins import int
from builtins import str
import argparse
import json
import os
import time
import numpy as np

import network
import random
import load

MAX_EPOCHS = 100


def get_folder_name(start_time, experiment_name):
    folder_name = FOLDER_TO_SAVE + experiment_name + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def get_filename_for_saving(start_time, experiment_name):
    saved_filename = get_folder_name(start_time, experiment_name) + \
        "/{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5"
    return saved_filename


def plot_model(model, start_time, experiment_name):
    from keras.utils.visualize_util import plot
    plot(
        model,
        to_file=get_folder_name(start_time, experiment_name) + '/model.png',
        show_shapes=True,
        show_layer_names=False)


def save_params(params, start_time, experiment_name):
    saving_filename = get_folder_name(start_time, experiment_name) + \
        "/params.json"
    save_str = json.dumps(params, ensure_ascii=False)
    save_str = save_str if isinstance(save_str, str) \
        else save_str.decode('utf-8')
    with open(saving_filename, 'w') as outfile:
        outfile.write(save_str)


def get_augment_fn(params):
    if params.get("amplitude_scale", False):
        return amplitude_scale
    else:
        return None


def amplitude_scale(x):
    scales = np.random.uniform(low=0.25, high=2.0,
                               size=(x.shape[0], 1, 1))
    return scales * x


def data_generator(x, y, batch_size, augmenter=None):
    num_examples = x.shape[0]
    indices = np.arange(num_examples)
    while True:
        np.random.shuffle(indices)
        end = num_examples - batch_size + 1
        for i in range(0, end, batch_size):
            idx = indices[i:i+batch_size]
            x_mb = x[idx, ...]
            y_mb = y[idx, ...]
            if augmenter is not None:
                x_mb = augmenter(x_mb)
            yield (x_mb, y_mb)


def train(args, params):
    global FOLDER_TO_SAVE

    if args.overfit is True:
        params["overfit"] = True
        params["gaussian_noise"] = 0
        for key in params:
            if "dropout" in key:
                params[key] = 0

    params["test_split_start"] = args.test_split_start        

    dl, processor = load.load_train(params)

    x_train = dl.x_train
    y_train = dl.y_train
    print("Training size: " + str(len(x_train)) + " examples.")

    x_test = dl.x_test
    y_test = dl.y_test
    print("Test size: " + str(len(x_test)) + " examples.")

    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    experiment_name = args.experiment

    FOLDER_TO_SAVE = params["FOLDER_TO_SAVE"]
    params["EXPERIMENT_NAME"] = experiment_name

    save_params(params, start_time, experiment_name)

    params.update({
        "input_shape": x_train[0].shape,
        "num_categories": len(processor.classes)
    })

    model = network.build_network(**params)

    try:
        plot_model(model, start_time, experiment_name)
    except:
        print("Skipping plot")

    if args.overfit is True:
        monitor_metric = 'loss'
    else:
        monitor_metric = 'val_loss'

    from keras.callbacks import EarlyStopping
    stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=params.get('early_stop_patience', 8),
        verbose=args.verbose)

    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=params.get('reduce_lr_factor', 0.1),
        patience=params.get('reduce_lr_patience', 2),
        min_lr=params["learning_rate"] * 0.001,
        verbose=args.verbose)

    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(start_time, experiment_name),
        save_best_only=False,
        verbose=args.verbose)

    batch_size = params.get("batch_size", 32)

    train_data = data_generator(
                    x_train, y_train,
                    batch_size=batch_size,
                    augmenter=get_augment_fn(params))

    samples_per_epoch = batch_size * int(x_train.shape[0] / batch_size)

    model.fit_generator(
        train_data,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=MAX_EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[checkpointer, reduce_lr, stopping],
        verbose=args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    parser.add_argument("--verbose", "-v", help="verbosity level", default=1,
                        type=int)
    parser.add_argument("--test_split_start", "-t", help="test split start",
                        default=0, type=int)
    parser.add_argument(
        "--overfit",
        help="whether to overfit training set",
        action="store_true")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
