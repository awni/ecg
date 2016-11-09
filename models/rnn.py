from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl

class RNN:

    def init_inference(self, config):
        self.output_dim = num_labels = config['output_dim']
        self.batch_size = batch_size = config['batch_size']

        self.inputs = inputs = tf.placeholder(tf.float32, shape=(batch_size, None))
        acts = tf.reshape(inputs, (batch_size, 1, -1, 1)) #NHWC

        acts = tfl.convolution2d(acts, num_outputs=32, kernel_size=[1, 8], stride=4)
        #acts = tfl.max_pool2d(acts, kernel_size=[1, 2], padding='SAME')

        acts = tfl.convolution2d(acts, num_outputs=64, kernel_size=[1, 8], stride=4)
        #acts = tfl.max_pool2d(acts, kernel_size=[1, 2], padding='SAME')

        acts = tfl.convolution2d(acts, num_outputs=128, kernel_size=[1, 8], stride=4)
        #acts = tfl.max_pool2d(acts, kernel_size=[1, 2], padding='SAME')

        acts = tf.squeeze(tf.reduce_mean(acts, reduction_indices=2))
        acts = tfl.fully_connected(acts, 256)
        self.logits = tfl.fully_connected(acts, self.output_dim)

    def init_loss(self):
        self.labels = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   self.logits, self.labels))
        correct = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    def init_train(self, config):
        learning_rate = config['learning_rate']
        self.momentum = tf.Variable(config['momentum'],
                                    trainable=False,
                                    dtype=tf.float32)
        ema = tf.train.ExponentialMovingAverage(0.95)
        ema_op = ema.apply([self.loss, self.acc])
        self.avg_loss = ema.average(self.loss)
        self.avg_acc = ema.average(self.acc)

        tf.scalar_summary("Loss", self.loss)
        tf.scalar_summary("Accuracy", self.acc)

        self.it = tf.Variable(0, trainable=False, dtype=tf.int64)
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)
        train_op = optimizer.minimize(self.loss, global_step=self.it)
        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(ema_op)

    def set_momentum(self):
        self.momentum.assign(0.999)

    def feed_dict(self, inputs, labels=None):
        """
        Generates a feed dictionary for the model's place-holders.
        Params:
            inputs : List of 1D arrays of wave segments
            labels (optional) : List of integer labels
        Returns:
            feed_dict (use with feed_dict kwarg in session.run)
        """
        seq_lens = [i.shape[0] for i in inputs]
        batch_size = self.batch_size
        input_mat = np.zeros((batch_size, max(seq_lens)), dtype=np.float32)
        for b in range(batch_size):
            input_mat[b, :seq_lens[b]] = inputs[b]
        return {self.inputs : input_mat, self.labels : labels}
