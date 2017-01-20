def add_conv_layers(model, **params):
    from keras.layers.convolutional import Convolution1D
    from keras.regularizers import l2
    from keras.layers import Dropout
    subsample_lengths = params["conv_subsample_lengths"]
    for subsample_length in subsample_lengths:
        model.add(Convolution1D(
            nb_filter=params["conv_num_filters"],
            filter_length=params["conv_filter_length"],
            border_mode='same',
            subsample_length=subsample_length,
            init=params["conv_init"],
            activation=params["conv_activation"],
            W_regularizer=l2(params["conv_l2_penalty"])))
        if "conv_dropout" in params and params["conv_dropout"] > 0:
            model.add(Dropout(params["conv_dropout"]))


def add_recurrent_layers(model, **params):
    from keras.layers.recurrent import LSTM, GRU
    from keras.layers.wrappers import Bidirectional
    for i in range(params["recurrent_layers"]):
        rt = params["recurrent_type"]
        if rt == 'GRU':
            Recurrent = GRU
        elif rt == 'LSTM':
            Recurrent = LSTM
        rec_layer = Recurrent(
                    params["recurrent_hidden"],
                    consume_less="gpu",
                    dropout_W=params["recurrent_dropout"],
                    dropout_U=params["recurrent_dropout"],
                    return_sequences=True)
        if params["recurrent_is_bidirectional"] is True:
            model.add(Bidirectional(rec_layer))
        else:
            model.add(rec_layer)


def add_dense_layers(model, **params):
    from keras.layers.core import Dense
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Dropout
    from keras.regularizers import l2
    for i in range(params["dense_layers"]):
        model.add(TimeDistributed(Dense(
            params["dense_hidden"],
            activation=params["dense_activation"],
            init=params["dense_init"],
            W_regularizer=l2(params["dense_l2_penalty"]))))
        if "dense_dropout" in params and params["dense_dropout"] > 0:
            model.add(Dropout(params["dense_dropout"]))


def add_output_layer(model, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    model.add(TimeDistributed(Dense(params["num_categories"])))
    model.add(Activation('softmax'))


def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(lr=params["learning_rate"])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


def build_network(**params):
    from keras.models import Sequential
    from keras.layers import Reshape
    model = Sequential()
    model.add(Reshape(
        params["input_shape"],
        input_shape=params["input_shape"]))
    add_conv_layers(model, **params)
    add_recurrent_layers(model, **params)
    add_dense_layers(model, **params)
    add_output_layer(model, **params)
    add_compile(model, **params)
    return model
