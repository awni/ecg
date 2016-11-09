
import tensorflow as tf
import tensorflow.contrib.layers as tfl

import model

class CNN(model.Model):

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


