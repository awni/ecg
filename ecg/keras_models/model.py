
def build_network(**params):
    from keras.layers.core import Activation, Dense
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM, GRU
    from keras.layers.convolutional import Convolution1D
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Dropout
    from keras.layers.wrappers import Bidirectional

    subsample_lengths = params["conv_subsample_lengths"]
    model = Sequential()
    for subsample_length in subsample_lengths:
        model.add(Convolution1D(
            nb_filter=params["conv_num_filters"],
            filter_length=params["conv_filter_length"],
            border_mode='same',
            subsample_length=subsample_length,
            input_shape=params["input_shape"],
            init=params["conv_init"],
            activation=params["conv_activation"]))
        if "dropout" in params and params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))

    for i in range(params["recurrent_layers"]):
        rt = params["recurrent_type"]
        if rt == 'GRU':
            Recurrent = GRU
        elif rt == 'LSTM':
            Recurrent = LSTM
        rec_layer = Recurrent(
                    params["recurrent_hidden"],
                    return_sequences=True)
        if params["recurrent_is_bidirectional"] == True:
            model.add(Bidirectional(rec_layer))
        else:
            model.add(rec_layer)

    for i in range(params["dense_layers"]):
        model.add(TimeDistributed(Dense(
            params["dense_hidden"],
            activation=params["dense_activation"],
            init=params["dense_init"])))
        if "dropout" in params and params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))

    model.add(TimeDistributed(Dense(params["num_categories"])))
    model.add(Activation('softmax'))

    from keras.optimizers import Adam
    optimizer = Adam(lr=params["learning_rate"])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
