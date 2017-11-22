import tensorflow as tf

from utils import device_or_none, get_conv2d_kernel_size
from initializers.xavier import xavier_initializer_conv2d, xavier_constant_conv2d

def conv2d(
        x, kernel_size, out_channels, 
        strides=[1,1,1,1],
        padding='SAME',
        weight_filter=None,
        with_bias=False,
        bias_initializer=tf.zeros_initializer,
        activation=None,
        variable_device=None,
        reuse=False,
        name=None,
        **kwargs
        ):

    scope_name = name or 'conv2d'

    ks = get_conv2d_kernel_size(kernel_size)
    
    with tf.variable_scope(None, default_name=scope_name, reuse=reuse):
        xs = x.get_shape().as_list()

        with device_or_none(variable_device):
            weight = tf.get_variable(
                shape=[ks[0], ks[1], xs[3], out_channels],
                initializer=xavier_initializer_conv2d(x, ks[0], ks[1]),
                name='weight')

        weight = weight_filter(weight) if weight_filter else weight

        y = tf.nn.conv2d(x, weight, strides=strides, padding=padding)

        if with_bias:
            with device_or_none(variable_device):
                bias = tf.get_variable(
                    shape=[out_channels], initializer=bias_initializer(), name='bias')

            y = tf.nn.bias_add(out, bias)


        if activation:
            y = activation(y)

        return y

def equalized_conv2d(
        x, kernel_size, out_channels, 
        strides=[1,1,1,1],
        padding='SAME',
        weight_filter=None,
        with_bias=False,
        bias_initializer=tf.zeros_initializer,
        activation=None,
        variable_device=None,
        reuse=False,
        name=None,
        **kwargs
        ):

    scope_name = name or 'equalized_conv2d'

    ks = get_conv2d_kernel_size(kernel_size)
    
    with tf.variable_scope(None, default_name=scope_name, reuse=reuse):
        xs = x.get_shape().as_list()

        with device_or_none(variable_device):
            weight = tf.get_variable(
                shape=[ks[0], ks[1], xs[3], out_channels],
                initializer=tf.truncated_normal_initializer(),
                name='weight')

        weight = weight * xavier_constant_conv2d(x, ks[0], ks[1])

        y = tf.nn.conv2d(x, weight, strides=strides, padding=padding)

        if with_bias:
            with device_or_none(variable_device):
                bias = tf.get_variable(
                    shape=[out_channels], initializer=bias_initializer(), name='bias')

            y = tf.nn.bias_add(out, bias)

        if activation:
            y = activation(y)

        return y

