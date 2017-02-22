from keras import backend as K


def _bn_relu(layer, dropout=0, **params):
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

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer


def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers.convolutional import Convolution1D
    layer = Convolution1D(
        nb_filter=num_filters,
        filter_length=filter_length,
        border_mode='same',
        subsample_length=subsample_length,
        init=params["conv_init"])(layer)
    return layer


def resnet_block(
        layer,
        num_filters,
        subsample_length,
        zero_pad=False,
        **params):
    from keras.layers import merge
    from keras.layers.pooling import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_length=subsample_length)(layer)
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["num_skip"]):
        layer = _bn_relu(
            layer,
            dropout=params["conv_dropout"] if i > 0 else 0,
            **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = merge([shortcut, layer], mode="sum")
    return layer


def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / 2) * num_start_filters


def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"])
        zero_pad = (index % 2) == 0 and index > 0
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            zero_pad=zero_pad,
            **params)
    layer = _bn_relu(layer, **params)
    return layer


def add_conv_layers(layer, **params):
    from keras.layers import merge
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"])
        shortcut = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length,
            **params)
        layer = shortcut
        for i in range(params["num_skip"]):
            layer = _bn_relu(layer, dropout=params["conv_dropout"], **params)
            layer = add_conv_weight(
                layer,
                params["conv_filter_length"],
                num_filters,
                subsample_length=1,
                **params)
        layer = merge([shortcut, layer], mode="sum")
    return layer


def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)


def add_compile(model, **params):
    if params["optimizer"] == "adam":
        from keras.optimizers import Adam
        optimizer = Adam(
            lr=params["learning_rate"],
            clipnorm=params.get("clipnorm", 1))
    else:
        assert(params["optimizer"] == 'sgd')
        from keras.optimizers import SGD
        optimizer = SGD(
            lr=params["learning_rate"],
            decay=params.get("decay", 1e-4),
            momentum=params.get("momentum", 0.9),
            clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    if params.get("is_correct_resnet", False) is True:
        layer = add_resnet_layers(inputs, **params)
    else:
        layer = add_conv_layers(inputs, **params)
    output = add_output_layer(layer, **params)
    model = Model(input=[inputs], output=[output])
    add_compile(model, **params)
    return model
