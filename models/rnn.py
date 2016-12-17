
import tensorflow as tf
import tensorflow.contrib.layers as tfl

import model

class RNN(model.Model):

    def init_inference(self, config):
        self.output_dim = num_labels = config['output_dim']
        self.batch_size = batch_size = config['batch_size']

        self.inputs = inputs = tf.placeholder(tf.float32, shape=(batch_size, None))
        acts = tf.reshape(inputs, (batch_size, -1, 1, 1))

        for layer in config['conv_layers']:
            num_filters = layer['num_filters']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            acts = tfl.convolution2d(acts, num_outputs=num_filters,
                                     kernel_size=[kernel_size, 1],
                                     stride=stride)

        # Activations should emerge from the convolution with shape
        # [batch_size, time (subsampled), 1, num_channels]
        acts = tf.squeeze(acts, squeeze_dims=[2])

        bidirectional = config['rnn'].get('bidirectional', False)
        rnn_dim = config['rnn']['dim']
        cell_type = config['rnn'].get('cell_type', 'gru')
        if bidirectional:
            acts = _bi_rnn(acts, rnn_dim, cell_type)
        else:
            acts = _rnn(acts, rnn_dim, cell_type)

        self.logits = tfl.fully_connected(acts, self.output_dim)
        self.probs = tf.nn.softmax(self.logits)

def _rnn(acts, input_dim, cell_type, scope=None):
    if cell_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(input_dim)
    elif cell_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(input_dim)
    else:
        msg = "Invalid cell type {}".format(cell_type)
        raise ValueError(msg)

    acts, _ = tf.nn.dynamic_rnn(cell, acts,
                  dtype=tf.float32, scope=scope)
    return acts

def _bi_rnn(acts, input_dim, cell_type):
    """
    For some reason tf.bidirectional_dynamic_rnn requires a sequence length.
    """
    # Forwards
    with tf.variable_scope("fw") as fw_scope:
        acts_fw = _rnn(acts, input_dim, cell_type,
                       scope=fw_scope)

    # Backwards
    with tf.variable_scope("bw") as bw_scope:
        reverse_dims = [False, True, False]
        acts_bw = tf.reverse(acts, dims=reverse_dims)
        acts_bw = _rnn(acts_bw, input_dim, cell_type,
                       scope=bw_scope)
        acts_bw = tf.reverse(acts_bw, dims=reverse_dims)

    # Sum the forward and backward states.
    return tf.add(acts_fw, acts_bw)

