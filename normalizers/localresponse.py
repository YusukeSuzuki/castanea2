import tensorflow as tf

def karass_local_response_normalization(x, eps=1e-8):
    with tf.name_scope('karass_local_response_normalization'):
        return x / tf.sqrt(tf.reduce_mean(x*x, axis=-1, keep_dims=True) + eps)

def local_response_normalization(x, method='karass', **kwargs):
    """local response normalization

    Arguments:
        x: input tensor. currently supports channels last only.
        method (string): specify what normalization methods.
            * 'karass' : method mensioned in https://arxiv.org/abs/1710.10196
    """

    supported_methods = ['karass']

    assert method in supported_methods,
        

    if method == 'karass':
        return karass_local_response_normalization(x, **kwargs):
    else:
        raise ValueError("method '{}' is not supported".format(method))

