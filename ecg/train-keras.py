import argparse
import numpy as np
import random

from loader import Loader

def create_model(input_shape, num_categories):
    from keras.layers.core import Activation, Dense
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM
    from keras.layers.convolutional import Convolution1D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.layers.wrappers import TimeDistributed
    subsample_lengths = [2, 2, 2, 5, 5]
    model = Sequential()
    for subsample_length in subsample_lengths:
        model.add(Convolution1D(
            32, 32, # number of filters should be high, filter_length should at least be subsample length
            border_mode='same', 
            subsample_length=subsample_length,
            input_shape=input_shape,
            init='he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
    model.add(
        LSTM(
            100,
            return_sequences=True
        )
    )
    model.add(TimeDistributed(Dense(100, activation='relu', init='he_normal')))
    model.add(TimeDistributed(Dense(num_categories)))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    random.seed(20)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument("--refresh", help="whether to refresh cache")
    args = parser.parse_args()
    batch_size = 32
    dl = Loader(
        args.data_path,
        batch_size, use_one_hot_labels=True,
        use_cached_if_available=not args.refresh)

    x_train = dl.x_train[:, :, np.newaxis]
    y_train = dl.y_train
    print("Training size: " + str(len(x_train)) + " examples.")

    x_val = dl.x_test[:, :, np.newaxis]
    y_val = dl.y_test
    print("Validation size: " + str(len(x_val)) + " examples.")

    model = create_model(x_train[0].shape, dl.output_dim)

    try:
        from keras.utils.visualize_util import plot
        plot(model, to_file='model.png', show_shapes=True)
    except:
        pass

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=20,
              validation_data=(x_val, y_val))
