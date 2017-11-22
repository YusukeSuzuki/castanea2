import tensorflow as tf
from utils import get_conv2d_kernel_size

def _initializer_constant(size, uniform=False):
    s = 6. if uniform else 2.
    return tf.sqrt(s/size)

def _initializer(size, uniform=False, span=None, seed=None):
    stddev = _initializer_constant(size, uniform)

    span = span or [-1., 1.]

    assert span[0] < span[1], 'span must be span[0] < span[1].'

    if uniform:
        return tf.random_uniform_initializer(
            minval=span[0]*stddev, maxval=span[1]*stddev, seed=seed)
    else:
        mean = (span[0] * span[1]) / 2.
        scale = tf.abs(span[1] - mean)
        return tf.truncated_normal_initializer(
            mean=mean, stddev=scale*stddev, seed=seed)


def xavier_initializer_dense(x, uniform=False, span=None, seed=None):
    return _initializer(
        x.get_shape().as_list()[1], uniform, span, seed)


def xavier_initializer_conv2d(x, kernel_size, uniform=False, span=None, seed=None):
    ks = get_conv2d_kernel_size(kernel_size)

    return _initializer(
        ks[0] * ks[1] * x.get_shape().as_list()[3], uniform, span, seed)


def xavier_constant_dense(x, uniform=False, span=None):
    return _initializer_constant(
        x.get_shape().as_list()[1], uniform)


def xavier_constant_conv2d(x, kernel_size, uniform=False, span=None):
    ks = get_conv2d_kernel_size(kernel_size)
    return _initializer_constant(
        ks[0] * ks[1] * x.get_shape().as_list()[3], uniform)


