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

import load
from models import nn

MAX_EPOCHS = 500


def get_folder_name(start_time, net_type):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def get_filename_for_saving(start_time, net_type):
    saved_filename = get_folder_name(start_time, net_type) + \
        "/{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5"
    return saved_filename


def plot_model(model, start_time, net_type):
    from keras.utils.visualize_util import plot
    plot(
        model,
        to_file=get_folder_name(start_time, net_type) + '/model.png',
        show_shapes=True,
        show_layer_names=False)


def save_params(params, start_time, net_type):
    saving_filename = get_folder_name(start_time, net_type) + "/params.json"
    save_str = json.dumps(params, ensure_ascii=False)
    save_str = save_str if isinstance(save_str, str) \
        else save_str.decode('utf-8')
    with open(saving_filename, 'w') as outfile:
        outfile.write(save_str)


def train(args, params):
    global FOLDER_TO_SAVE
    # if overfit, remove all dropout
    if args.overfit is True:
        params["overfit"] = True
        for key in params:
            if "dropout" in key:
                params[key] = 0

    dl = load.load(args, params)

    x_train = dl.x_train
    y_train = dl.y_train
    print("Training size: " + str(len(x_train)) + " examples.")

    x_val = dl.x_test
    y_val = dl.y_test
    print("Validation size: " + str(len(x_val)) + " examples.")

    start_time = str(int(time.time()))

    FOLDER_TO_SAVE = params["FOLDER_TO_SAVE"]

    net_type = str(params["version"])

    save_params(params, start_time, net_type)

    params.update({
        "input_shape": x_train[0].shape,
        "num_categories": dl.output_dim
    })

    model = nn.build_network(**params)

    try:
        plot_model(model, start_time, net_type)
    except:
        print("Skipping plot")

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from keras.callbacks import EarlyStopping

    if args.overfit is True:
        monitor_metric = 'loss'
    else:
        monitor_metric = 'val_loss'

    stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=20,
        verbose=args.verbose)

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=args.verbose)

    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(start_time, net_type),
        save_best_only=False,
        verbose=args.verbose)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        nb_epoch=MAX_EPOCHS,
        callbacks=[checkpointer, reduce_lr, stopping],
        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data files")
    parser.add_argument("config_file", help="path to confile file")
    parser.add_argument("--verbose", "-v", help="verbosity level", default=2)
    parser.add_argument(
        "--overfit",
        help="whether to overfit training set",
        action="store_true")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
