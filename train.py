from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os
import random
import tensorflow as tf
import time

from models import rnn
import loader

tf.flags.DEFINE_string("config", "mitdb_config.json",
                       "Configuration file for training.")
FLAGS = tf.flags.FLAGS

def run_epoch(model, data_loader, session):
    for batch in data_loader.batches(data_loader.train):
        ops = [model.train_op, model.avg_loss, model.avg_acc, model.it]
        res = session.run(ops, feed_dict=model.feed_dict(*batch))
        _, loss, acc, it = res
        if it % 100 == 0:
            msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
            print(msg.format(it, loss, acc))

def main(argv=None):

    with open(FLAGS.config) as fid:
        config = json.load(fid)

    random.seed(config['seed'])
    epochs = config['optimizer']['epochs']
    data_loader = loader.Loader(config['data']['path'],
                                config['model']['batch_size'])
    model = rnn.RNN()

    save_path = config['io']['save_path']
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    config['model']['output_dim'] = data_loader.vocab_size
    with open(os.path.join(save_path, "config.json"), 'w') as fid:
        json.dump(config, fid)

    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(config['seed'])
        mean, std = data_loader.mean_and_std()
        model.init_inference(config['model'], mean=mean, std=std)
        model.init_loss()
        model.init_train(config['optimizer'])
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in range(epochs):
            start = time.time()
            run_epoch(model, data_loader, sess)
            saver.save(sess, os.path.join(save_path, "model"))
            print("Epoch {} time {:.1f} (s)".format(e, time.time() - start))

if __name__ == '__main__':
    tf.app.run()
