
import tensorflow as tf

def make_summary(tag, value):
    value = tf.Summary.Value(tag=tag, simple_value=value)
    return tf.Summary(value=[value])
