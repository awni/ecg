from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os
import tensorflow as tf

import loader
import models
import decoder

tf.flags.DEFINE_string("save_path", None,
                       "Path to saved model.")
FLAGS = tf.flags.FLAGS

class Evaler:

    def __init__(self, save_path, batch_size=1):
        config_file = os.path.join(save_path, "config.json")

        with open(config_file, 'r') as fid:
            config = json.load(fid)
        config['model']['batch_size'] = batch_size

        self.model = getattr(models, config['model']['model_class'])()
        self.graph = tf.Graph()
        self.session = sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model.init_inference(config['model'])
            tf.global_variables_initializer().run(session=sess)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, os.path.join(save_path, "model"))

    def probs(self, inputs):
        model = self.model
        probs, = self.session.run([model.probs], model.feed_dict(inputs))
        return probs

    def predict(self, inputs):
        probs = self.probs(inputs)
        return np.argmax(probs, axis=2)

def main(argv=None):
    assert FLAGS.save_path is not None, \
        "Must provide the path to a model directory."

    config_file = os.path.join(FLAGS.save_path, "config.json")
    with open(config_file, 'r') as fid:
        config = json.load(fid)

    batch_size = 32
    data_loader = loader.Loader(config['data']['path'], batch_size,
                                seed=config['data']['seed'])

    lm = decoder.LM(data_loader, order=2)
    evaler = Evaler(FLAGS.save_path, batch_size=batch_size)

    corr = 0.0
    total = 0
    for inputs, labels in data_loader.batches(data_loader.val):
        probs = evaler.probs(inputs)
        predictions = [decoder.beam_search(probs[i,...], lm)
                         for i in range(batch_size)]
        predictions = np.vstack(predictions)
        corr += np.sum(predictions == np.vstack(labels))
        total += predictions.size
    print("Number {}, Accuracy {:.3f}".format(total, corr / total))


if __name__ == "__main__":
    tf.app.run()
