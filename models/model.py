
import numpy as np
import tensorflow as tf

MOMENTUM_INIT = 0.5

class Model:

    def init_inference(self, config):
        raise NotImplemented("Your model must implement this function.")

    def init_loss(self):
        self.labels = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   self.logits, self.labels))
        correct = tf.equal(tf.argmax(self.logits, 1), self.labels)
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    def init_train(self, config):
        self.momentum = config['momentum']
        self.mom_var = tf.Variable(MOMENTUM_INIT, trainable=False,
                                   dtype=tf.float32)
        ema = tf.train.ExponentialMovingAverage(0.95)
        ema_op = ema.apply([self.loss, self.acc])
        self.avg_loss = ema.average(self.loss)
        self.avg_acc = ema.average(self.acc)

        tf.scalar_summary("Loss", self.loss)
        tf.scalar_summary("Accuracy", self.acc)

        self.it = tf.Variable(0, trainable=False, dtype=tf.int64)

        learning_rate = tf.train.exponential_decay(config['learning_rate'],
                            self.it, config['decay_steps'],
                            config['decay_rate'], staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate, self.mom_var)
        train_op = optimizer.minimize(self.loss, global_step=self.it)
        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(ema_op)

    def set_momentum(self, session):
        self.mom_var.assign(self.momentum).eval(session=session)

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

