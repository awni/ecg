
import tensorflow as tf
import tensorflow.contrib.layers as tfl

import model

class RNN(model.Model):

    def init_inference(self, config):
        self.output_dim = num_labels = config['output_dim']
        self.batch_size = batch_size = config['batch_size']

        self.inputs = inputs = tf.placeholder(tf.float32, shape=(batch_size, None))
        acts = tf.reshape(inputs, (batch_size, -1, 1, 1))

        num_filters = [32, 64, 128]

        for nf in num_filters:
            acts = tfl.convolution2d(acts, num_outputs=nf,
                                     kernel_size=[8, 1], stride=4)
            #acts = tfl.max_pool2d(acts, kernel_size=[2, 1],
            #                      padding='SAME')

        # Activations should emerge from the convolution with shape
        # [batch_size, time (subsampled), 1, num_channels]
        acts = tf.squeeze(acts, squeeze_dims=[2])

        cell = tf.nn.rnn_cell.GRUCell(128)
        acts, _ = tf.nn.dynamic_rnn(cell, acts, dtype=tf.float32)

        acts = tf.reduce_mean(acts, reduction_indices=1)
        acts = tfl.fully_connected(acts, 256)
        self.logits = tfl.fully_connected(acts, self.output_dim)


