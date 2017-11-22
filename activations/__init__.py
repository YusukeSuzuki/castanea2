import tensorflow as tf


def swish(x, beta=1.):
    return x * tf.nn.sigmoid(x*beta)


def lrelu(x, beta=1e-2):
    return tf.maximum(x * beta, x)


def pseud_tanh(x, beta1=1e-2, beta2=1e-0):
    return tf.minimum(beta1*(x+beta2),tf.maximum(beta1*(x-beta2), x))


def pseud_sigmoid(x, beta1=1e-2, beta2=1e-0):
    return tf.minimum(beta1*(x+beta2),tf.maximum(beta1*(x-beta2), x)) + 1.


def scaled_tanh(x, beta1, beta2=1., shift=0.):
    return tf.tanh(x*beta2) * beta1 + shift


