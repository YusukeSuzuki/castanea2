import tensorflow as tf


def karras_minibatch_statistics(x):
    with tf.name_scope('karras_minibatch_statistics'):
        xs = x.get_shape().as_list()
        m, v = tf.nn.moments(x, axes=[0,3], keep_dims=True)
        avg_sdev = tf.reduce_mean(tf.sqrt(v), axis=3, keep_dims=True)
        avg_sdev = tf.tile(avg_sdev, [xs[0], 1, 1, 1])
    return avg_sdev

