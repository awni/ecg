
def build_network(**params):
    from keras.layers.core import Activation, Dense
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM, GRU
    from keras.layers.convolutional import Convolution1D
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Dropout

    subsample_lengths = params["subsample_lengths"]
    model = Sequential()
    for subsample_length in subsample_lengths:
        model.add(Convolution1D(
            nb_filter=params["num_filters"],
            filter_length=params["filter_length"],
            border_mode='same', 
            subsample_length=subsample_length,
            input_shape=params["input_shape"],
            init='he_normal',
            activation='relu'))
        if "dropout" in params and params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))

    for i in range(params["recurrent_layers"]):
        model.add(
            GRU(
                params["recurrent_hidden"],
                return_sequences=True
            )
        )

    for i in range(params["dense_layers"]):
        model.add(TimeDistributed(Dense(
            params["dense_hidden"],
            activation='relu',
            init='he_normal')))
        if "dropout" in params and params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))

    model.add(TimeDistributed(Dense(params["num_categories"])))
    model.add(Activation('softmax'))

    from keras.optimizers import Adam
    optimizer = Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
