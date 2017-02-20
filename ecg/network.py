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


def add_conv_weight(layer, filter_length, subsample_length, **params):
    from keras.layers.convolutional import Convolution1D
    layer = Convolution1D(
        nb_filter=params["conv_num_filters"],
        filter_length=filter_length,
        border_mode='same',
        subsample_length=subsample_length,
        init=params["conv_init"])(layer)
    return layer


def resnet_block(layer, subsample_length, **params):
    from keras.layers import merge
    from keras.layers.pooling import MaxPooling1D

    shortcut = MaxPooling1D(pool_length=subsample_length)(layer)

    for i in range(params["num_skip"]):
        layer = _bn_relu(layer, **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            subsample_length if i == 0 else 1,
            **params)

    layer = merge([shortcut, layer], mode="sum")
    return layer


def add_resnet_layers(layer, **params):
    layer = _bn_relu(add_conv_weight(layer, 16, 1, **params), **params)
    for subsample_length in params["conv_subsample_lengths"]:
        layer = resnet_block(layer, subsample_length, **params)
    layer = _bn_relu(layer, **params)
    return layer


def add_conv_layers(layer, **params):
    from keras.layers import merge
    from keras.layers.noise import GaussianNoise
    for subsample_length in params["conv_subsample_lengths"]:
        if params.get("gaussian_noise", 0) > 0:
            layer = GaussianNoise(params["gaussian_noise"])(layer)
        shortcut = add_conv_weight(
            layer,
            params["conv_filter_length"],
            subsample_length,
            **params)
        layer = shortcut
        for i in range(params["num_skip"]):
            layer = _bn_relu(layer, **params)
            layer = add_conv_weight(
                layer,
                params["conv_filter_length"],
                1,
                **params)
        layer = merge([shortcut, layer], mode="sum")
    return layer


def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)


def add_compile(model, **params):
    from keras.optimizers import SGD
    optimizer = SGD(
        lr=params["learning_rate"], decay=params["decay"],
        momentum=params["momentum"], clipnorm=params["clipnorm"])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    if (params.get("is_correct_resnet", False)):
        layer = add_resnet_layers(inputs, **params)
    else:
        layer = add_conv_layers(inputs, **params)
    output = add_output_layer(layer, **params)
    model = Model(input=[inputs], output=[output])
    add_compile(model, **params)
    return model
