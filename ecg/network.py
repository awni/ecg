def _bn_relu(layer, **params):
    activation_fn = params["conv_activation"]

    if activation_fn == 'prelu':
        from keras.layers.advanced_activations import PReLU
        layer = PReLU()(layer)
    elif activation_fn == 'elu':
        from keras.layers.advanced_activations import ELU
        layer = ELU()(layer)
    elif activation_fn == 'leaky_relu':
        from keras.layers.advanced_activations import LeakyReLU
        layer = LeakyReLU()(layer)
    else:
        from keras.layers import BatchNormalization
        from keras.layers import Activation
        layer = BatchNormalization()(layer)
        layer = Activation(activation_fn)(layer)

    if params.get("conv_dropout", 0) > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer


def add_conv_layers(layer, **params):
    from keras.layers.convolutional import Convolution1D
    from keras.layers import merge
    from keras.layers.noise import GaussianNoise

    def add_conv_weight(layer, subsample_length):
        layer = Convolution1D(
            nb_filter=params["conv_num_filters"],
            filter_length=params["conv_filter_length"],
            border_mode='same',
            subsample_length=subsample_length,
            init=params["conv_init"])(layer)
        return layer

    subsample_lengths = params["conv_subsample_lengths"]
    for subsample_length in subsample_lengths:
        if params.get("gaussian_noise", 0) > 0:
            layer = GaussianNoise(params["gaussian_noise"])(layer)
        shortcut = add_conv_weight(layer, subsample_length)

        layer = shortcut

        if params.get("is_resnet", True):
            for i in range(params["num_skip"]):
                layer = _bn_relu(layer, **params)
                layer = add_conv_weight(layer, 1)

            layer = merge([shortcut, layer], mode="sum")
        else:
            layer = _bn_relu(layer, **params)

    return layer


def add_recurrent_layers(layer, **params):
    from keras.layers.recurrent import LSTM, GRU
    from keras.layers.wrappers import Bidirectional
    for i in range(params.get("recurrent_layers", 0)):
        rt = params["recurrent_type"]
        if rt == 'GRU':
            Recurrent = GRU
        elif rt == 'LSTM':
            Recurrent = LSTM
        rec_layer = Recurrent(
                    params["recurrent_hidden"],
                    dropout_W=params["recurrent_dropout"],
                    dropout_U=params["recurrent_dropout"],
                    return_sequences=True)
        if params["recurrent_is_bidirectional"] is True:
            layer = Bidirectional(rec_layer)(layer)
        else:
            layer = rec_layer(layer)
    return layer


def add_dense_layers(layer, **params):
    from keras.layers.core import Dense
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Dropout
    from keras.regularizers import l2
    for i in range(params.get("dense_layers", 0)):
        layer = TimeDistributed(Dense(
            params["dense_hidden"],
            activation=params["dense_activation"],
            init=params["dense_init"],
            W_regularizer=l2(params["dense_l2_penalty"])))(layer)
        if params.get("dense_dropout", 0) > 0:
            layer = Dropout(params["dense_dropout"])(layer)
    return layer


def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)


def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(lr=params["learning_rate"], clipnorm=params["clipnorm"])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    layer = add_conv_layers(inputs, **params)
    layer = add_recurrent_layers(layer, **params)
    layer = add_dense_layers(layer, **params)
    output = add_output_layer(layer, **params)
    model = Model(input=[inputs], output=[output])
    add_compile(model, **params)
    return model
