import argparse
import numpy as np

from loader import Loader

def create_model(input_shape, num_categories):
    from keras.layers.core import Activation, Dense
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM
    from keras.layers.convolutional import Convolution1D
    from keras.layers.wrappers import TimeDistributed

    model = Sequential()
    model.add(Convolution1D(
        10, 100,
        border_mode='same', subsample_length=200, input_shape=input_shape))
    model.add(
        LSTM(
            32,
            return_sequences=True
        )
    )
    model.add(TimeDistributed(Dense(num_categories)))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    from keras.utils.visualize_util import plot
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

    print(x_train[0].shape, y_train.shape, y_train[0].shape)
    model = create_model(x_train[0].shape, dl.output_dim)
    plot(model, to_file='model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=5,
          validation_data=(x_val, y_val))
